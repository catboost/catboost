set(_GNU_COMMON_C_CXX_FLAGS "\
  -fexceptions \
  -fno-common \
  -fcolor-diagnostics \
  -faligned-allocation \
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
  -D_GNU_SOURCE \
  -D__LONG_LONG_SUPPORTED \
  -D_YNDX_LIBUNWIND_ENABLE_EXCEPTION_BACKTRACE \
  -D_libunwind_ \
  -DLIBCXX_BUILDING_LIBCXXRT \
")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${_GNU_COMMON_C_CXX_FLAGS} \
  -D_FILE_OFFSET_BITS=64 \
")
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

if (NOT APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fuse-init-array")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fuse-init-array")
endif()

if (ANDROID)
  include_directories(SYSTEM ${CMAKE_ANDROID_NDK}/sources/cxx-stl/llvm-libc++abi/include)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsigned-char")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsigned-char")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FILE_OFFSET_BITS=64")
endif()

if (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsigned-char")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsigned-char")
endif()

if ((CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64") OR (CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64"))
  set(_X86_64_GNU_COMPILER_FLAGS "\
    -m64 \
    -msse2 \
    -msse3 \
    -mssse3 \
    -msse4.1 \
    -msse4.2 \
    -mpopcnt \
    -mcx16 \
  ")

  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${_X86_64_GNU_COMPILER_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_X86_64_GNU_COMPILER_FLAGS}")
endif()
