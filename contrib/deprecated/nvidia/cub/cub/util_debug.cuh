/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2022, NVIDIA CORPORATION.  All rights reserved.
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
 * Error and event logging routines.
 *
 * The following macros definitions are supported:
 * - \p CUB_LOG.  Simple event messages are printed to \p stdout.
 */

#pragma once
#pragma clang system_header


#include <cub/util_namespace.cuh>
#include <cub/util_arch.cuh>

#include <nv/target>

#include <cstdio>

CUB_NAMESPACE_BEGIN


#ifdef DOXYGEN_SHOULD_SKIP_THIS // Only parse this during doxygen passes:

/**
 * @def CUB_DEBUG_LOG
 *
 * Causes kernel launch configurations to be printed to the console
 */
#define CUB_DEBUG_LOG

/**
 * @def CUB_DEBUG_SYNC
 *
 * Causes synchronization of the stream after every kernel launch to check 
 * for errors. Also causes kernel launch configurations to be printed to the 
 * console.
 */
#define CUB_DEBUG_SYNC

/**
 * @def CUB_DEBUG_HOST_ASSERTIONS
 *
 * Extends `CUB_DEBUG_SYNC` effects by checking host-side precondition 
 * assertions.
 */
#define CUB_DEBUG_HOST_ASSERTIONS

/**
 * @def CUB_DEBUG_DEVICE_ASSERTIONS
 *
 * Extends `CUB_DEBUG_HOST_ASSERTIONS` effects by checking device-side 
 * precondition assertions.
 */
#define CUB_DEBUG_DEVICE_ASSERTIONS

/**
 * @def CUB_DEBUG_ALL
 *
 * Causes host and device-side precondition assertions to be checked. Apart 
 * from that, causes synchronization of the stream after every kernel launch to 
 * check for errors. Also causes kernel launch configurations to be printed to 
 * the console.
 */
#define CUB_DEBUG_ALL

#endif // DOXYGEN_SHOULD_SKIP_THIS 

/**
 * \addtogroup UtilMgmt
 * @{
 */


// `CUB_DETAIL_DEBUG_LEVEL_*`: Implementation details, internal use only:

#define CUB_DETAIL_DEBUG_LEVEL_NONE 0
#define CUB_DETAIL_DEBUG_LEVEL_HOST_ASSERTIONS_ONLY 1
#define CUB_DETAIL_DEBUG_LEVEL_LOG 2
#define CUB_DETAIL_DEBUG_LEVEL_SYNC 3
#define CUB_DETAIL_DEBUG_LEVEL_HOST_ASSERTIONS 4
#define CUB_DETAIL_DEBUG_LEVEL_DEVICE_ASSERTIONS 5
#define CUB_DETAIL_DEBUG_LEVEL_ALL 1000

// `CUB_DEBUG_*`: User interfaces:

// Extra logging, no syncs
#ifdef CUB_DEBUG_LOG
#define CUB_DETAIL_DEBUG_LEVEL CUB_DETAIL_DEBUG_LEVEL_LOG
#endif

// Logging + syncs
#ifdef CUB_DEBUG_SYNC
#define CUB_DETAIL_DEBUG_LEVEL CUB_DETAIL_DEBUG_LEVEL_SYNC
#endif

// Logging + syncs + host assertions
#ifdef CUB_DEBUG_HOST_ASSERTIONS
#define CUB_DETAIL_DEBUG_LEVEL CUB_DETAIL_DEBUG_LEVEL_HOST_ASSERTIONS
#endif

// Logging + syncs + host assertions + device assertions
#ifdef CUB_DEBUG_DEVICE_ASSERTIONS
#define CUB_DETAIL_DEBUG_LEVEL CUB_DETAIL_DEBUG_LEVEL_DEVICE_ASSERTIONS
#endif

// All
#ifdef CUB_DEBUG_ALL
#define CUB_DETAIL_DEBUG_LEVEL CUB_DETAIL_DEBUG_LEVEL_ALL 
#endif

// Default case, no extra debugging:
#ifndef CUB_DETAIL_DEBUG_LEVEL
#ifdef NDEBUG
#define CUB_DETAIL_DEBUG_LEVEL CUB_DETAIL_DEBUG_LEVEL_NONE
#else
#define CUB_DETAIL_DEBUG_LEVEL CUB_DETAIL_DEBUG_LEVEL_HOST_ASSERTIONS_ONLY
#endif
#endif

/*
 * `CUB_DETAIL_DEBUG_ENABLE_*`:
 * Internal implementation details, used for testing enabled debug features:
 */

#if CUB_DETAIL_DEBUG_LEVEL >= CUB_DETAIL_DEBUG_LEVEL_LOG
#define CUB_DETAIL_DEBUG_ENABLE_LOG
#endif

#if CUB_DETAIL_DEBUG_LEVEL >= CUB_DETAIL_DEBUG_LEVEL_SYNC
#define CUB_DETAIL_DEBUG_ENABLE_SYNC
#endif

#if (CUB_DETAIL_DEBUG_LEVEL >= CUB_DETAIL_DEBUG_LEVEL_HOST_ASSERTIONS) || \
    (CUB_DETAIL_DEBUG_LEVEL == CUB_DETAIL_DEBUG_LEVEL_HOST_ASSERTIONS_ONLY)
#define CUB_DETAIL_DEBUG_ENABLE_HOST_ASSERTIONS
#endif

#if CUB_DETAIL_DEBUG_LEVEL >= CUB_DETAIL_DEBUG_LEVEL_DEVICE_ASSERTIONS
#define CUB_DETAIL_DEBUG_ENABLE_DEVICE_ASSERTIONS
#endif


/// CUB error reporting macro (prints error messages to stderr)
#if (defined(DEBUG) || defined(_DEBUG)) && !defined(CUB_STDERR)
    #define CUB_STDERR
#endif

/**
 * \brief %If \p CUB_STDERR is defined and \p error is not \p cudaSuccess, the
 * corresponding error message is printed to \p stderr (or \p stdout in device
 * code) along with the supplied source context.
 *
 * \return The CUDA error.
 */
__host__ __device__
__forceinline__
cudaError_t Debug(cudaError_t error, const char *filename, int line)
{
  // Clear the global CUDA error state which may have been set by the last
  // call. Otherwise, errors may "leak" to unrelated kernel launches.

  // clang-format off
  #ifndef CUB_RDC_ENABLED
  #define CUB_TEMP_DEVICE_CODE
  #else
  #define CUB_TEMP_DEVICE_CODE cudaGetLastError()
  #endif

  NV_IF_TARGET(
    NV_IS_HOST, 
    (cudaGetLastError();),
    (CUB_TEMP_DEVICE_CODE;)
  );
  
  #undef CUB_TEMP_DEVICE_CODE
  // clang-format on

#ifdef CUB_STDERR
  if (error)
  {
    NV_IF_TARGET(
      NV_IS_HOST, (
        fprintf(stderr,
                "CUDA error %d [%s, %d]: %s\n",
                error,
                filename,
                line,
                cudaGetErrorString(error));
        fflush(stderr);
      ),
      (
        printf("CUDA error %d [block (%d,%d,%d) thread (%d,%d,%d), %s, %d]\n",
               error,
               blockIdx.z,
               blockIdx.y,
               blockIdx.x,
               threadIdx.z,
               threadIdx.y,
               threadIdx.x,
               filename,
               line);
      )
    );
  }
#else
  (void)filename;
  (void)line;
#endif

  return error;
}

/**
 * \brief Debug macro
 */
#ifndef CubDebug
    #define CubDebug(e) CUB_NS_QUALIFIER::Debug((cudaError_t) (e), __FILE__, __LINE__)
#endif


/**
 * \brief Debug macro with exit
 */
#ifndef CubDebugExit
    #define CubDebugExit(e) if (CUB_NS_QUALIFIER::Debug((cudaError_t) (e), __FILE__, __LINE__)) { exit(1); }
#endif


/**
 * \brief Log macro for printf statements.
 */
#if !defined(_CubLog)
#if defined(_NVHPC_CUDA) || !(defined(__clang__) && defined(__CUDA__))

// NVCC / NVC++
#define _CubLog(format, ...)                                                   \
  do                                                                           \
  {                                                                            \
    NV_IF_TARGET(NV_IS_HOST,                                                   \
                 (printf(format, __VA_ARGS__);),                               \
                 (printf("[block (%d,%d,%d), thread (%d,%d,%d)]: " format,     \
                         blockIdx.z,                                           \
                         blockIdx.y,                                           \
                         blockIdx.x,                                           \
                         threadIdx.z,                                          \
                         threadIdx.y,                                          \
                         threadIdx.x,                                          \
                         __VA_ARGS__);));                                      \
  } while (false)

#else // Clang:

// XXX shameless hack for clang around variadic printf...
//     Compilies w/o supplying -std=c++11 but shows warning,
//     so we silence them :)
#pragma clang diagnostic ignored "-Wc++11-extensions"
#pragma clang diagnostic ignored "-Wunnamed-type-template-args"
template <class... Args>
inline __host__ __device__ void va_printf(char const *format,
                                          Args const &...args)
{
#ifdef __CUDA_ARCH__
  printf(format,
         blockIdx.z,
         blockIdx.y,
         blockIdx.x,
         threadIdx.z,
         threadIdx.y,
         threadIdx.x,
         args...);
#else
  printf(format, args...);
#endif
}
#ifndef __CUDA_ARCH__
#define _CubLog(format, ...) CUB_NS_QUALIFIER::va_printf(format, __VA_ARGS__);
#else
#define _CubLog(format, ...)                                                   \
  CUB_NS_QUALIFIER::va_printf("[block (%d,%d,%d), thread "                     \
                              "(%d,%d,%d)]: " format,                          \
                              __VA_ARGS__);
#endif
#endif
#endif

/** @} */       // end group UtilMgmt

CUB_NAMESPACE_END
