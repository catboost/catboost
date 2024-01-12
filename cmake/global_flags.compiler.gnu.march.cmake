
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
elseif (ANDROID AND (CMAKE_ANDROID_ARCH_ABI STREQUAL "armeabi-v7a"))
  string(APPEND _GNU_MARCH_C_CXX_FLAGS " -mfloat-abi=softfp")
endif()
