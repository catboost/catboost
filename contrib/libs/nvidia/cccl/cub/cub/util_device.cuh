/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2020, NVIDIA CORPORATION.  All rights reserved.
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

//! \file
//! Properties of a given CUDA device and the corresponding PTX bundle.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/util_debug.cuh>
#include <cub/util_policy_wrapper_t.cuh>
#include <cub/util_type.cuh>
// for backward compatibility
#include <cub/util_temporary_storage.cuh>

#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/array>
#include <cuda/std/atomic>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#if !_CCCL_COMPILER(NVRTC)
#  if defined(CUB_DEFINE_RUNTIME_POLICIES)
#    include <format>
#    include <string_view>

#    error #include <nlohmann/json.hpp>
#  endif // defined(CUB_DEFINE_RUNTIME_POLICIES)
#endif // !_CCCL_COMPILER(NVRTC)

#if defined(CUB_ENABLE_POLICY_PTX_JSON)
#  include <cub/detail/ptx-json/json.h>
#endif // defined(CUB_ENABLE_POLICY_PTX_JSON)

#include <nv/target>

CUB_NAMESPACE_BEGIN

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

namespace detail
{
/**
 * \brief Empty kernel for querying PTX manifest metadata (e.g., version) for the current device
 */
template <typename T>
CUB_DETAIL_KERNEL_ATTRIBUTES void EmptyKernel()
{}

} // namespace detail

#endif // _CCCL_DOXYGEN_INVOKED

#if !_CCCL_COMPILER(NVRTC)
/**
 * \brief Returns the current device or -1 if an error occurred.
 */
CUB_RUNTIME_FUNCTION inline int CurrentDevice()
{
  int device = -1;
  if (CubDebug(cudaGetDevice(&device)))
  {
    return -1;
  }
  return device;
}

#  ifndef _CCCL_DOXYGEN_INVOKED // Do not document

//! @brief RAII helper which saves the current device and switches to the specified device on construction and switches
//! to the saved device on destruction.
class SwitchDevice
{
  int target_device_;
  int original_device_;

public:
  //! @brief Queries the current device and if that is different than @p target_device sets the current device to
  //! @p target_device
  SwitchDevice(const int target_device)
      : target_device_(target_device)
  {
    CubDebug(cudaGetDevice(&original_device_));
    if (original_device_ != target_device_)
    {
      CubDebug(cudaSetDevice(target_device_));
    }
  }

  //! @brief If the @p original_device was not equal to @p target_device sets the current device back to
  //! @p original_device
  ~SwitchDevice()
  {
    if (original_device_ != target_device_)
    {
      CubDebug(cudaSetDevice(original_device_));
    }
  }
};

#  endif // _CCCL_DOXYGEN_INVOKED

/**
 * \brief Returns the number of CUDA devices available or -1 if an error
 *        occurred.
 */
CUB_RUNTIME_FUNCTION inline int DeviceCountUncached()
{
  int count = -1;
  if (CubDebug(cudaGetDeviceCount(&count)))
  {
    // CUDA makes no guarantees about the state of the output parameter if
    // `cudaGetDeviceCount` fails; in practice, they don't, but out of
    // paranoia we'll reset `count` to `-1`.
    count = -1;
  }
  return count;
}

// Host code. This is a separate function to avoid defining a local static in a host/device function.
_CCCL_HOST inline int DeviceCountCachedValue()
{
  static int count = DeviceCountUncached();
  return count;
}

/**
 * \brief Returns the number of CUDA devices available.
 *
 * \note This function may cache the result internally.
 *
 * \note This function is thread safe.
 */
CUB_RUNTIME_FUNCTION inline int DeviceCount()
{
  int result = -1;

  NV_IF_TARGET(NV_IS_HOST, (result = DeviceCountCachedValue();), (result = DeviceCountUncached();));

  return result;
}

#  ifndef _CCCL_DOXYGEN_INVOKED // Do not document
/**
 * \brief Per-device cache for a CUDA attribute value; the attribute is queried
 *        and stored for each device upon construction.
 */
struct PerDeviceAttributeCache
{
  struct DevicePayload
  {
    int attribute;
    cudaError_t error;
  };

  // Each entry starts in the `DeviceEntryEmpty` state, then proceeds to the
  // `DeviceEntryInitializing` state, and then proceeds to the
  // `DeviceEntryReady` state. These are the only state transitions allowed;
  // i.e. a linear sequence of transitions.
  enum DeviceEntryStatus
  {
    DeviceEntryEmpty = 0,
    DeviceEntryInitializing,
    DeviceEntryReady
  };

  struct DeviceEntry
  {
    ::cuda::std::atomic<DeviceEntryStatus> flag;
    DevicePayload payload;
  };

private:
  ::cuda::std::array<DeviceEntry, detail::max_devices> entries_;

public:
  /**
   * \brief Construct the cache.
   */
  _CCCL_HOST inline PerDeviceAttributeCache()
      : entries_()
  {
    assert(DeviceCount() <= detail::max_devices);
  }

  /**
   * \brief Retrieves the payload of the cached function \p f for \p device.
   *
   * \note You must pass a morally equivalent function in to every call or
   *       this function has undefined behavior.
   */
  template <typename Invocable>
  _CCCL_HOST DevicePayload operator()(Invocable&& f, int device)
  {
    if (device >= DeviceCount() || device < 0)
    {
      return DevicePayload{0, cudaErrorInvalidDevice};
    }

    auto& entry   = entries_[device];
    auto& flag    = entry.flag;
    auto& payload = entry.payload;

    DeviceEntryStatus old_status = DeviceEntryEmpty;

    // First, check for the common case of the entry being ready.
    if (flag.load(::cuda::std::memory_order_acquire) != DeviceEntryReady)
    {
      // Assume the entry is empty and attempt to lock it so we can fill
      // it by trying to set the state from `DeviceEntryReady` to
      // `DeviceEntryInitializing`.
      if (flag.compare_exchange_strong(
            old_status, DeviceEntryInitializing, ::cuda::std::memory_order_acq_rel, ::cuda::std::memory_order_acquire))
      {
        // We successfully set the state to `DeviceEntryInitializing`;
        // we have the lock and it's our job to initialize this entry
        // and then release it.

        // We don't use `CubDebug` here because we let the user code
        // decide whether or not errors are hard errors.
        payload.error = ::cuda::std::forward<Invocable>(f)(payload.attribute);
        if (payload.error)
        {
          // Clear the global CUDA error state which may have been
          // set by the last call. Otherwise, errors may "leak" to
          // unrelated kernel launches.
          cudaGetLastError();
        }

        // Release the lock by setting the state to `DeviceEntryReady`.
        flag.store(DeviceEntryReady, ::cuda::std::memory_order_release);
      }

      // If the `compare_exchange_weak` failed, then `old_status` has
      // been updated with the value of `flag` that it observed.

      else if (old_status == DeviceEntryInitializing)
      {
        // Another execution agent is initializing this entry; we need
        // to wait for them to finish; we'll know they're done when we
        // observe the entry status as `DeviceEntryReady`.
        do
        {
          old_status = flag.load(::cuda::std::memory_order_acquire);
        } while (old_status != DeviceEntryReady);
        // FIXME: Use `atomic::wait` instead when we have access to
        // host-side C++20 atomics. We could use libcu++, but it only
        // supports atomics for SM60 and up, even if you're only using
        // them in host code.
      }
    }

    // We now know that the state of our entry is `DeviceEntryReady`, so
    // just return the entry's payload.
    return entry.payload;
  }
};
#  endif // _CCCL_DOXYGEN_INVOKED

/**
 * \brief Retrieves the PTX version that will be used on the current device (major * 100 + minor * 10).
 */
CUB_RUNTIME_FUNCTION inline cudaError_t PtxVersionUncached(int& ptx_version)
{
  // Instantiate `EmptyKernel<void>` in both host and device code to ensure
  // it can be called.
  using EmptyKernelPtr                         = void (*)();
  [[maybe_unused]] EmptyKernelPtr empty_kernel = detail::EmptyKernel<void>;

  // Define a temporary macro that expands to the current target ptx version
  // in device code.
  // <nv/target> may provide an abstraction for this eventually. For now,
  // we have to keep this usage of __CUDA_ARCH__.
#  if _CCCL_CUDA_COMPILER(NVHPC)
#    define CUB_TEMP_GET_PTX __builtin_current_device_sm()
#  else // ^^^ _CCCL_CUDA_COMPILER(NVHPC) ^^^ / vvv !_CCCL_CUDA_COMPILER(NVHPC) vvv
#    define CUB_TEMP_GET_PTX _CCCL_PTX_ARCH()
#  endif // ^^^ !_CCCL_CUDA_COMPILER(NVHPC) ^^^

  cudaError_t result = cudaSuccess;
  NV_IF_TARGET(
    NV_IS_HOST,
    (cudaFuncAttributes empty_kernel_attrs;

     result = CubDebug(cudaFuncGetAttributes(&empty_kernel_attrs, reinterpret_cast<void*>(empty_kernel)));

     ptx_version = empty_kernel_attrs.ptxVersion * 10;),
    // NV_IS_DEVICE
    (
      // This is necessary to ensure instantiation of EmptyKernel in device
      // code. The `reinterpret_cast` is necessary to suppress a
      // set-but-unused warnings. This is a meme now:
      // https://twitter.com/blelbach/status/1222391615576100864
      (void) reinterpret_cast<EmptyKernelPtr>(empty_kernel);

      ptx_version = CUB_TEMP_GET_PTX;));

#  undef CUB_TEMP_GET_PTX

  return result;
}

/**
 * \brief Retrieves the PTX version that will be used on \p device (major * 100 + minor * 10).
 */
_CCCL_HOST inline cudaError_t PtxVersionUncached(int& ptx_version, int device)
{
  [[maybe_unused]] SwitchDevice sd(device);
  return PtxVersionUncached(ptx_version);
}

template <typename Tag>
_CCCL_HOST inline PerDeviceAttributeCache& GetPerDeviceAttributeCache()
{
  static PerDeviceAttributeCache cache;
  return cache;
}

struct PtxVersionCacheTag
{};
struct SmVersionCacheTag
{};

/**
 * \brief Retrieves the PTX virtual architecture that will be used on \p device (major * 100 + minor * 10). If
 * __CUDA_ARCH_LIST__ is defined, this value is one of __CUDA_ARCH_LIST__.
 *
 * \note This function may cache the result internally.
 * \note This function is thread safe.
 */
_CCCL_HOST inline cudaError_t PtxVersion(int& ptx_version, int device)
{
  // Note: the ChainedPolicy pruning (i.e., invoke_static) requites that there's an exact match between one of the
  // architectures in __CUDA_ARCH__ and the runtime queried ptx version.
  auto const payload = GetPerDeviceAttributeCache<PtxVersionCacheTag>()(
    // If this call fails, then we get the error code back in the payload, which we check with `CubDebug` below.
    [=](int& pv) {
      return PtxVersionUncached(pv, device);
    },
    device);

  if (!CubDebug(payload.error))
  {
    ptx_version = payload.attribute;
  }

  return payload.error;
}

/**
 * \brief Retrieves the PTX virtual architecture that will be used on the current device (major * 100 + minor * 10).
 *
 * \note This function may cache the result internally.
 * \note This function is thread safe.
 */
CUB_RUNTIME_FUNCTION inline cudaError_t PtxVersion(int& ptx_version)
{
  // Note: the ChainedPolicy pruning (i.e., invoke_static) requites that there's an exact match between one of the
  // architectures in __CUDA_ARCH__ and the runtime queried ptx version.
  cudaError_t result = cudaErrorUnknown;
  NV_IF_TARGET(NV_IS_HOST,
               (result = PtxVersion(ptx_version, CurrentDevice());),
               ( // NV_IS_DEVICE:
                 result = PtxVersionUncached(ptx_version);));
  return result;
}

/**
 * \brief Retrieves the SM version (i.e. compute capability) of \p device (major * 100 + minor * 10)
 */
CUB_RUNTIME_FUNCTION inline cudaError_t SmVersionUncached(int& sm_version, int device = CurrentDevice())
{
  cudaError_t error = cudaSuccess;
  do
  {
    int major = 0, minor = 0;
    error = CubDebug(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    if (cudaSuccess != error)
    {
      break;
    }

    error = CubDebug(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    if (cudaSuccess != error)
    {
      break;
    }
    sm_version = major * 100 + minor * 10;
  } while (0);

  return error;
}

/**
 * \brief Retrieves the SM version (i.e. compute capability) of \p device (major * 100 + minor * 10).
 *
 * \note This function may cache the result internally.
 * \note This function is thread safe.
 */
CUB_RUNTIME_FUNCTION inline cudaError_t SmVersion(int& sm_version, int device = CurrentDevice())
{
  cudaError_t result = cudaErrorUnknown;

  NV_IF_TARGET(
    NV_IS_HOST,
    (auto const payload = GetPerDeviceAttributeCache<SmVersionCacheTag>()(
       // If this call fails, then we get the error code back in the payload, which we check with `CubDebug` below.
       [=](int& pv) {
         return SmVersionUncached(pv, device);
       },
       device);

     if (!CubDebug(payload.error)) { sm_version = payload.attribute; };

     result = payload.error;),
    ( // NV_IS_DEVICE
      result = SmVersionUncached(sm_version, device);));

  return result;
}

//! Synchronize the specified \p stream when called in host code. Otherwise, does nothing.
CUB_RUNTIME_FUNCTION inline cudaError_t SyncStream([[maybe_unused]] cudaStream_t stream)
{
  NV_IF_TARGET(NV_IS_HOST, (return CubDebug(cudaStreamSynchronize(stream));), (return cudaErrorNotSupported;))
}

namespace detail
{
//! If CUB_DEBUG_SYNC is defined and this function is called from host code, a sync is performed and the
//! sync result is returned. Otherwise, does nothing.
CUB_RUNTIME_FUNCTION inline cudaError_t DebugSyncStream([[maybe_unused]] cudaStream_t stream)
{
#  ifdef CUB_DEBUG_SYNC
  NV_IF_TARGET(NV_IS_HOST,
               (_CubLog("%s", "Synchronizing...\n"); return SyncStream(stream);),
               (_CubLog("%s", "WARNING: Skipping CUB debug synchronization in device code"); return cudaSuccess;));
#  else // ^^^ CUB_DEBUG_SYNC / !CUB_DEBUG_SYNC vvv
  return cudaSuccess;
#  endif // ^^^ !CUB_DEBUG_SYNC ^^^
}

/** \brief Gets whether the current device supports unified addressing */
CUB_RUNTIME_FUNCTION inline cudaError_t HasUVA(bool& has_uva)
{
  has_uva           = false;
  int device        = -1;
  cudaError_t error = CubDebug(cudaGetDevice(&device));
  if (cudaSuccess != error)
  {
    return error;
  }

  int uva = 0;
  error   = CubDebug(cudaDeviceGetAttribute(&uva, cudaDevAttrUnifiedAddressing, device));
  if (cudaSuccess != error)
  {
    return error;
  }
  has_uva = uva == 1;
  return error;
}

} // namespace detail

/**
 * @brief Computes maximum SM occupancy in thread blocks for executing the given kernel function
 *        pointer @p kernel_ptr on the current device with @p block_threads per thread block.
 *
 * @par Snippet
 * The code snippet below illustrates the use of the MaxSmOccupancy function.
 * @par
 * @code
 * #include <cub/cub.cuh>   // or equivalently <cub/util_device.cuh>
 *
 * template <typename T>
 * __global__ void ExampleKernel()
 * {
 *     // Allocate shared memory for BlockScan
 *     __shared__ volatile T buffer[4096];
 *
 *        ...
 * }
 *
 *     ...
 *
 * // Determine SM occupancy for ExampleKernel specialized for unsigned char
 * int max_sm_occupancy;
 * MaxSmOccupancy(max_sm_occupancy, ExampleKernel<unsigned char>, 64);
 *
 * // max_sm_occupancy  <-- 4 on SM10
 * // max_sm_occupancy  <-- 8 on SM20
 * // max_sm_occupancy  <-- 12 on SM35
 *
 * @endcode
 *
 * @param[out] max_sm_occupancy
 *   maximum number of thread blocks that can reside on a single SM
 *
 * @param[in] kernel_ptr
 *   Kernel pointer for which to compute SM occupancy
 *
 * @param[in] block_threads
 *   Number of threads per thread block
 *
 * @param[in] dynamic_smem_bytes
 *   Dynamically allocated shared memory in bytes. Default is 0.
 */
template <typename KernelPtr>
_CCCL_VISIBILITY_HIDDEN CUB_RUNTIME_FUNCTION inline cudaError_t
MaxSmOccupancy(int& max_sm_occupancy, KernelPtr kernel_ptr, int block_threads, int dynamic_smem_bytes = 0)
{
  return CubDebug(
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_sm_occupancy, kernel_ptr, block_threads, dynamic_smem_bytes));
}

#endif // !_CCCL_COMPILER(NVRTC)

/******************************************************************************
 * Policy management
 ******************************************************************************/
// PolicyWrapper

namespace detail
{

#if defined(CUB_DEFINE_RUNTIME_POLICIES) || defined(CUB_ENABLE_POLICY_PTX_JSON)
#  if !_CCCL_HAS_CONCEPTS()
#    error Generation of runtime policy wrappers and/or policy PTX JSON information requires C++20 concepts.
#  endif // !_CCCL_HAS_CONCEPTS()
#endif // defined(CUB_DEFINE_RUNTIME_POLICIES) || defined(CUB_ENABLE_POLICY_PTX_JSON)

#define CUB_DETAIL_POLICY_WRAPPER_CONCEPT_TEST(field)     , StaticPolicyT::_CCCL_PP_FIRST field
#define CUB_DETAIL_POLICY_WRAPPER_REFINE_CONCEPT(concept) concept<StaticPolicyT>&&

#define CUB_DETAIL_POLICY_WRAPPER_ACCESSOR(field)                   \
  __host__ __device__ static constexpr auto _CCCL_PP_SECOND field() \
  {                                                                 \
    return StaticPolicyT::_CCCL_PP_FIRST field;                     \
  }

#if defined(CUB_ENABLE_POLICY_PTX_JSON)
#  define CUB_DETAIL_POLICY_WRAPPER_ENCODED_FIELD(field) \
    key<_CCCL_TO_STRING(_CCCL_PP_FIRST field)>() = value<(int) StaticPolicyT::_CCCL_PP_FIRST field>(),

#  define CUB_DETAIL_POLICY_WRAPPER_ENCODED_POLICY(...)                                     \
    _CCCL_DEVICE static constexpr auto EncodedPolicy()                                      \
    {                                                                                       \
      using namespace ptx_json;                                                             \
      return object<_CCCL_PP_FOR_EACH(CUB_DETAIL_POLICY_WRAPPER_ENCODED_FIELD, __VA_ARGS__) \
                      key<"__dummy">() = value<0>()>();                                     \
    }
#else
#  define CUB_DETAIL_POLICY_WRAPPER_ENCODED_POLICY(...)
#endif // defined(CUB_ENABLE_POLICY_PTX_JSON)

#if defined(CUB_DEFINE_RUNTIME_POLICIES)
#  define CUB_DETAIL_POLICY_WRAPPER_FIELD(field)                       \
    _CCCL_PP_THIRD field _CCCL_PP_CAT(runtime_, _CCCL_PP_FIRST field); \
    _CCCL_PP_THIRD field _CCCL_PP_SECOND field() const                 \
    {                                                                  \
      return _CCCL_PP_CAT(runtime_, _CCCL_PP_FIRST field);             \
    }

#  define CUB_DETAIL_POLICY_WRAPPER_GET_FIELD(field)  \
    ap._CCCL_PP_CAT(runtime_, _CCCL_PP_FIRST field) = \
      static_cast<_CCCL_PP_THIRD field>(subpolicy[_CCCL_TO_STRING(_CCCL_PP_FIRST field)].get<int>());

#  define CUB_DETAIL_POLICY_WRAPPER_FIELD_STRING(field) \
    _CCCL_TO_STRING(static constexpr auto _CCCL_PP_FIRST field = static_cast<_CCCL_PP_THIRD field>({});) "\n"

#  define CUB_DETAIL_POLICY_WRAPPER_FIELD_VALUE(field) , (int) ap._CCCL_PP_CAT(runtime_, _CCCL_PP_FIRST field)

#  define CUB_DETAIL_POLICY_WRAPPER_AGENT_POLICY(concept_name, ...)                                                    \
    struct Runtime##concept_name                                                                                       \
    {                                                                                                                  \
      _CCCL_PP_FOR_EACH(CUB_DETAIL_POLICY_WRAPPER_FIELD, __VA_ARGS__)                                                  \
      static std::pair<Runtime##concept_name, std::string>                                                             \
      from_json(const nlohmann::json& json, std::string_view subpolicy_name)                                           \
      {                                                                                                                \
        auto subpolicy = json[subpolicy_name];                                                                         \
        assert(!subpolicy.is_null());                                                                                  \
        Runtime##concept_name ap;                                                                                      \
        _CCCL_PP_FOR_EACH(CUB_DETAIL_POLICY_WRAPPER_GET_FIELD, __VA_ARGS__)                                            \
        return std::make_pair(                                                                                         \
          ap,                                                                                                          \
          std::format("struct {} {{\n" _CCCL_PP_FOR_EACH(CUB_DETAIL_POLICY_WRAPPER_FIELD_STRING, __VA_ARGS__) "}};\n", \
                      subpolicy_name _CCCL_PP_FOR_EACH(CUB_DETAIL_POLICY_WRAPPER_FIELD_VALUE, __VA_ARGS__)));          \
      }                                                                                                                \
    };
#else
#  define CUB_DETAIL_POLICY_WRAPPER_AGENT_POLICY(...)
#endif // defined(CUB_DEFINE_RUNTIME_POLICIES)

template <typename T>
_CCCL_CONCEPT always_true = true;

#define CUB_DETAIL_POLICY_WRAPPER_DEFINE(concept_name, refines, ...)                                                   \
  template <typename StaticPolicyT>                                                                                    \
  _CCCL_CONCEPT concept_name = _CCCL_PP_FOR_EACH(CUB_DETAIL_POLICY_WRAPPER_REFINE_CONCEPT, _CCCL_PP_EXPAND refines)    \
    _CCCL_REQUIRES_EXPR((StaticPolicyT))(true _CCCL_PP_FOR_EACH(CUB_DETAIL_POLICY_WRAPPER_CONCEPT_TEST, __VA_ARGS__)); \
  template <typename StaticPolicyT>                                                                                    \
  struct concept_name##Wrapper : StaticPolicyT                                                                         \
  {                                                                                                                    \
    __host__ __device__ constexpr concept_name##Wrapper(StaticPolicyT base)                                            \
        : StaticPolicyT(base)                                                                                          \
    {}                                                                                                                 \
    _CCCL_PP_FOR_EACH(CUB_DETAIL_POLICY_WRAPPER_ACCESSOR, __VA_ARGS__)                                                 \
    CUB_DETAIL_POLICY_WRAPPER_ENCODED_POLICY(__VA_ARGS__)                                                              \
  };                                                                                                                   \
  _CCCL_TEMPLATE(typename StaticPolicyT)                                                                               \
  _CCCL_REQUIRES(concept_name<StaticPolicyT>)                                                                          \
  __host__ __device__ constexpr concept_name##Wrapper<StaticPolicyT> MakePolicyWrapper(StaticPolicyT policy)           \
  {                                                                                                                    \
    return concept_name##Wrapper{policy};                                                                              \
  }                                                                                                                    \
  CUB_DETAIL_POLICY_WRAPPER_AGENT_POLICY(concept_name, __VA_ARGS__)

// Generic agent policy
CUB_DETAIL_POLICY_WRAPPER_DEFINE(
  GenericAgentPolicy, (always_true), (BLOCK_THREADS, BlockThreads, int), (ITEMS_PER_THREAD, ItemsPerThread, int) )

_CCCL_TEMPLATE(typename PolicyT)
_CCCL_REQUIRES((!GenericAgentPolicy<PolicyT>) )
__host__ __device__ constexpr PolicyT MakePolicyWrapper(PolicyT policy)
{
  return policy;
}

} // namespace detail

//----------------------------------------------------------------------------------------------------------------------
// ChainedPolicy

#if !_CCCL_COMPILER(NVRTC)

namespace detail
{

// Forward declaration of the default kernel launcher factory
struct TripleChevronFactory;

// By default, CUB uses `cub::detail::TripleChevronFactory` to access the CUDA runtime.
// The `CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY` indirection is used to override the default kernel launcher factory
// in CUB tests. This allows us to:
//   1. retrieve kernel pointers on the usage side of the API, and
//   2. validate use of specified CUDA stream by accelerated algorithms.
#  ifndef CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY
#    define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY cub::detail::TripleChevronFactory
#  endif

/**
 * Kernel dispatch configuration
 */
struct KernelConfig
{
  int block_threads{0};
  int items_per_thread{0};
  int tile_size{0};
  int sm_occupancy{0};

  template <typename AgentPolicyT,
            typename KernelPtrT,
            typename LauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t
  Init(KernelPtrT kernel_ptr, AgentPolicyT agent_policy = {}, LauncherFactory launcher_factory = {})
  {
    block_threads    = cub::detail::MakePolicyWrapper(agent_policy).BlockThreads();
    items_per_thread = cub::detail::MakePolicyWrapper(agent_policy).ItemsPerThread();
    tile_size        = block_threads * items_per_thread;
    return launcher_factory.MaxSmOccupancy(sm_occupancy, kernel_ptr, block_threads);
  }
};

} // namespace detail
#endif // !_CCCL_COMPILER(NVRTC)

/// Helper for dispatching into a policy chain
template <int PolicyPtxVersion, typename PolicyT, typename PrevPolicyT>
struct ChainedPolicy
{
  /// The policy for the active compiler pass
  using ActivePolicy = ::cuda::std::_If<(CUB_PTX_ARCH < PolicyPtxVersion), typename PrevPolicyT::ActivePolicy, PolicyT>;

#if !_CCCL_COMPILER(NVRTC)
  /// Specializes and dispatches op in accordance to the first policy in the chain of adequate PTX version
  template <typename FunctorT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Invoke(int device_ptx_version, FunctorT& op)
  {
    // __CUDA_ARCH_LIST__ is only available from CTK 11.5 onwards
#  if defined(__CUDA_ARCH_LIST__) && false
    return runtime_to_compiletime<1, __CUDA_ARCH_LIST__>(device_ptx_version, op);
    // NV_TARGET_SM_INTEGER_LIST is defined by NVHPC. The values need to be multiplied by 10 to match
    // __CUDA_ARCH_LIST__. E.g. arch 860 from __CUDA_ARCH_LIST__ corresponds to arch 86 from NV_TARGET_SM_INTEGER_LIST.
#  elif defined(NV_TARGET_SM_INTEGER_LIST) && false
    return runtime_to_compiletime<10, NV_TARGET_SM_INTEGER_LIST>(device_ptx_version, op);
#  else
    if (device_ptx_version < PolicyPtxVersion)
    {
      return PrevPolicyT::Invoke(device_ptx_version, op);
    }
    return op.template Invoke<PolicyT>();
#  endif
  }
#endif // !_CCCL_COMPILER(NVRTC)

private:
  template <int, typename, typename>
  friend struct ChainedPolicy; // let us call invoke_static of other ChainedPolicy instantiations

#if !_CCCL_COMPILER(NVRTC)
  template <int ArchMult, int... CudaArches, typename FunctorT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t runtime_to_compiletime(int device_ptx_version, FunctorT& op)
  {
    // We instantiate invoke_static for each CudaArches, but only call the one matching device_ptx_version.
    // If there's no exact match of the architectures in __CUDA_ARCH_LIST__/NV_TARGET_SM_INTEGER_LIST and the runtime
    // queried ptx version (i.e., the closest ptx version to the current device's architecture that the EmptyKernel was
    // compiled for), we return cudaErrorInvalidDeviceFunction. Such a scenario may arise if CUB_DISABLE_NAMESPACE_MAGIC
    // is set and different TUs are compiled for different sets of architecture.
    cudaError_t e = cudaErrorInvalidDeviceFunction;
    (..., (device_ptx_version == CudaArches * ArchMult ? (e = invoke_static<CudaArches * ArchMult>(op)) : cudaSuccess));
    return e;
  }

  template <int DevicePtxVersion, typename FunctorT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t invoke_static(FunctorT& op)
  {
    if constexpr (DevicePtxVersion < PolicyPtxVersion)
    {
      return PrevPolicyT::template invoke_static<DevicePtxVersion>(op);
    }
    else
    {
      return op.template Invoke<PolicyT>();
    }
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

/// Helper for dispatching into a policy chain (end-of-chain specialization)
template <int PolicyPtxVersion, typename PolicyT>
struct ChainedPolicy<PolicyPtxVersion, PolicyT, PolicyT>
{
  template <int, typename, typename>
  friend struct ChainedPolicy; // befriend primary template, so it can call invoke_static

  /// The policy for the active compiler pass
  using ActivePolicy = PolicyT;

#if !_CCCL_COMPILER(NVRTC)
  /// Specializes and dispatches op in accordance to the first policy in the chain of adequate PTX version
  template <typename FunctorT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Invoke(int /*ptx_version*/, FunctorT& op)
  {
    return op.template Invoke<PolicyT>();
  }

private:
  template <int, typename FunctorT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t invoke_static(FunctorT& op)
  {
    return op.template Invoke<PolicyT>();
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

CUB_NAMESPACE_END

#if _CCCL_HAS_CUDA_COMPILER() && !_CCCL_COMPILER(NVRTC)
#  include <cub/detail/launcher/cuda_runtime.cuh> // to complete the definition of TripleChevronFactory
#endif // _CCCL_HAS_CUDA_COMPILER() && !_CCCL_COMPILER(NVRTC)
