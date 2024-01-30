
set(_GNU_MARCH_C_CXX_FLAGS "")

if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(i686|x86_64|AMD64)$")
  if (CMAKE_SYSTEM_PROCESSOR STREQUAL "i686")
    string(APPEND _GNU_MARCH_C_CXX_FLAGS " -m32")
  elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    string(APPEND _GNU_MARCH_C_CXX_FLAGS " -m64")
  endif()
  string(APPEND _GNU_MARCH_C_CXX_FLAGS "\
    -msse2 \
    -msse3 \
    -mssse3 \
  ")

  if ((CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$") OR (NOT ANDROID))
    string(APPEND _GNU_MARCH_C_CXX_FLAGS "\
      -msse4.1 \
      -msse4.2 \
      -mpopcnt \
    ")
    if (NOT ANDROID)
      # older clang versions did not support this feature on Android:
      # https://reviews.llvm.org/rGc32d307a49f5255602e7543e64e6c38a7f536abc
      string(APPEND _GNU_MARCH_C_CXX_FLAGS " -mcx16")
    endif()
  endif()

  if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    string(APPEND _GNU_MARCH_C_CXX_FLAGS " -D_YNDX_LIBUNWIND_ENABLE_EXCEPTION_BACKTRACE")
  endif()
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
  if (CMAKE_SYSTEM_NAME MATCHES "^(Darwin|Linux)$")
    # Clang 13+ generates outline atomics by default if '-rtlib=compiler_rt' is specified or system's
    # libgcc version is >= 9.3.1 : https://github.com/llvm/llvm-project/commit/c5e7e649d537067dec7111f3de1430d0fc8a4d11
    # Disable this behaviour because our build links with contrib/libs/cxxsupp/builtins that does not contain outline atomics yet
    string(APPEND _GNU_MARCH_C_CXX_FLAGS " -mno-outline-atomics")
  endif()
elseif (ANDROID AND (CMAKE_ANDROID_ARCH_ABI STREQUAL "armeabi-v7a"))
  string(APPEND _GNU_MARCH_C_CXX_FLAGS " -mfloat-abi=softfp")
endif()
