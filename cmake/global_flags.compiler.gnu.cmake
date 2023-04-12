set(_GNU_COMMON_C_CXX_FLAGS "\
  -fexceptions \
  -fno-common \
  -fcolor-diagnostics \
  -faligned-allocation \
  -fdebug-default-version=4 \
  -ffunction-sections \
  -fdata-sections \
  -Wall \
  -Wextra \
  -Wno-parentheses \
  -Wno-implicit-const-int-float-conversion \
  -Wno-unknown-warning-option \
  -pipe \
  -D_THREAD_SAFE \
  -D_PTHREADS \
  -D_REENTRANT \
  -D_LIBCPP_ENABLE_CXX17_REMOVED_FEATURES \
  -D_LARGEFILE_SOURCE \
  -D__STDC_CONSTANT_MACROS \
  -D__STDC_FORMAT_MACROS \
  -D__LONG_LONG_SUPPORTED \
")

if (CMAKE_SYSTEM_NAME MATCHES "^(Android|Linux)$")
  string(APPEND _GNU_COMMON_C_CXX_FLAGS " -D_GNU_SOURCE")
endif()

if (CMAKE_SYSTEM_NAME MATCHES "^(Darwin|Linux)$")
  string(APPEND _GNU_COMMON_C_CXX_FLAGS " -DLIBCXX_BUILDING_LIBCXXRT")
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # Use .init_array instead of .ctors (default for old clang versions)
  # See: https://maskray.me/blog/2021-11-07-init-ctors-init-array
  string(APPEND _GNU_COMMON_C_CXX_FLAGS " -fuse-init-array")
endif()

if (ANDROID)
  include_directories(SYSTEM ${CMAKE_ANDROID_NDK}/sources/cxx-stl/llvm-libc++abi/include)

  # There is no usable _FILE_OFFSET_BITS=64 support in Androids until API 21. And it's incomplete until at least API 24.
  # https://android.googlesource.com/platform/bionic/+/master/docs/32-bit-abi.md
else()
  string(APPEND _GNU_COMMON_C_CXX_FLAGS " -D_FILE_OFFSET_BITS=64")
endif()

if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm.*|aarch64)")
  string(APPEND _GNU_COMMON_C_CXX_FLAGS " -fsigned-char")
endif()

if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(i686|x86_64|AMD64)$")
  if (CMAKE_SYSTEM_PROCESSOR STREQUAL "i686")
    string(APPEND _GNU_COMMON_C_CXX_FLAGS " -m32")
  elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    string(APPEND _GNU_COMMON_C_CXX_FLAGS " -m64")
  endif()
  string(APPEND _GNU_COMMON_C_CXX_FLAGS "\
    -msse2 \
    -msse3 \
    -mssse3 \
  ")

  if ((CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$") OR (NOT ANDROID))
    string(APPEND _GNU_COMMON_C_CXX_FLAGS "\
      -msse4.1 \
      -msse4.2 \
      -mpopcnt \
    ")
    if (NOT ANDROID)
      # older clang versions did not support this feature on Android:
      # https://reviews.llvm.org/rGc32d307a49f5255602e7543e64e6c38a7f536abc
      string(APPEND _GNU_COMMON_C_CXX_FLAGS " -mcx16")
    endif()
  endif()

  if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    string(APPEND _GNU_COMMON_C_CXX_FLAGS " -D_YNDX_LIBUNWIND_ENABLE_EXCEPTION_BACKTRACE")
  endif()
elseif (ANDROID AND (CMAKE_ANDROID_ARCH_ABI STREQUAL "armeabi-v7a"))
  string(APPEND _GNU_COMMON_C_CXX_FLAGS " -mfloat-abi=softfp")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${_GNU_COMMON_C_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_GNU_COMMON_C_CXX_FLAGS} \
  -Woverloaded-virtual \
  -Wimport-preprocessor-directive-pedantic \
  -Wno-undefined-var-template \
  -Wno-return-std-move \
  -Wno-defaulted-function-deleted \
  -Wno-pessimizing-move \
  -Wno-deprecated-anon-enum-enum-conversion \
  -Wno-deprecated-enum-enum-conversion \
  -Wno-deprecated-enum-float-conversion \
  -Wno-ambiguous-reversed-operator \
  -Wno-deprecated-volatile \
")
