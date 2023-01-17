
# This file was gererated by the build system used internally in the Yandex monorepo.
# Only simple modifications are allowed (adding source-files to targets, adding simple properties
# like target_include_directories). These modifications will be ported to original
# ya.make files by maintainers. Any complex modifications which can't be ported back to the
# original buildsystem will not be accepted.


cmake_minimum_required(VERSION 3.15)
project(CATBOOST LANGUAGES ASM C CXX)

set(BUILD_SHARED_LIBS Off)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS Off)
set(CATBOOST_MAX_LINK_JOBS 5 CACHE STRING "Maximum parallel link jobs for Ninja generator")
set_property(GLOBAL PROPERTY JOB_POOLS link_jobs=${CATBOOST_MAX_LINK_JOBS})
set(CMAKE_JOB_POOL_LINK link_jobs)
set(CMAKE_C_FLAGS "\
  -m64 \
  -fexceptions \
  -fno-common \
  -fuse-init-array \
  -fcolor-diagnostics \
  -faligned-allocation \
  -ffunction-sections \
  -fdata-sections \
  -Wall \
  -Wextra \
  -Wno-parentheses \
  -Wno-implicit-const-int-float-conversion \
  -Wno-unknown-warning-option \
  -D_THREAD_SAFE \
  -D_PTHREADS \
  -D_REENTRANT \
  -D_LARGEFILE_SOURCE \
  -D__STDC_CONSTANT_MACROS \
  -D__STDC_FORMAT_MACROS \
  -D_FILE_OFFSET_BITS=64 \
  -D_GNU_SOURCE \
  -D_YNDX_LIBUNWIND_ENABLE_EXCEPTION_BACKTRACE \
  -D__LONG_LONG_SUPPORTED \
  -DSSE_ENABLED=1 \
  -DSSE3_ENABLED=1 \
  -DSSSE3_ENABLED=1 \
  -DSSE41_ENABLED=1 \
  -DSSE42_ENABLED=1 \
  -DPOPCNT_ENABLED=1 \
  -DCX16_ENABLED=1 \
  -D_libunwind_ \
  -DLIBCXX_BUILDING_LIBCXXRT \
  -msse2 \
  -msse3 \
  -mssse3 \
  -msse4.1 \
  -msse4.2 \
  -mpopcnt \
  -mcx16 \
  "
)
set(CMAKE_CXX_FLAGS "\
  -m64 \
  -fexceptions \
  -fno-common \
  -fuse-init-array \
  -fcolor-diagnostics \
  -faligned-allocation \
  -ffunction-sections \
  -fdata-sections \
  -Wall \
  -Wextra \
  -Wno-parentheses \
  -Wno-implicit-const-int-float-conversion \
  -Wno-unknown-warning-option \
  -D_THREAD_SAFE \
  -D_PTHREADS \
  -D_REENTRANT \
  -D_LARGEFILE_SOURCE \
  -D__STDC_CONSTANT_MACROS \
  -D__STDC_FORMAT_MACROS \
  -D_FILE_OFFSET_BITS=64 \
  -D_GNU_SOURCE \
  -D_YNDX_LIBUNWIND_ENABLE_EXCEPTION_BACKTRACE \
  -D__LONG_LONG_SUPPORTED \
  -DSSE_ENABLED=1 \
  -DSSE3_ENABLED=1 \
  -DSSSE3_ENABLED=1 \
  -DSSE41_ENABLED=1 \
  -DSSE42_ENABLED=1 \
  -DPOPCNT_ENABLED=1 \
  -DCX16_ENABLED=1 \
  -D_libunwind_ \
  -DLIBCXX_BUILDING_LIBCXXRT \
  -msse2 \
  -msse3 \
  -mssse3 \
  -msse4.1 \
  -msse4.2 \
  -mpopcnt \
  -mcx16 \
  -Woverloaded-virtual \
  -Wimport-preprocessor-directive-pedantic \
  -Wno-undefined-var-template \
  -Wno-return-std-move \
  -Wno-address-of-packed-member \
  -Wno-defaulted-function-deleted \
  -Wno-pessimizing-move \
  -Wno-range-loop-construct \
  -Wno-deprecated-anon-enum-enum-conversion \
  -Wno-deprecated-enum-enum-conversion \
  -Wno-deprecated-enum-float-conversion \
  -Wno-ambiguous-reversed-operator \
  -Wno-deprecated-volatile \
  "
)
add_link_options(
  -nodefaultlibs
  -lc
  -lm
)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}/cmake)
include_directories(${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")
enable_testing()
include(cmake/archive.cmake)
include(cmake/conan.cmake)
include(cmake/cython.cmake)
include(cmake/fat_object.cmake)
include(cmake/fbs.cmake)
include(cmake/protobuf.cmake)
include(cmake/swig.cmake)
include(cmake/yandex_common.cmake)
include(cmake/global_vars.cmake)

conan_cmake_configure(
  GENERATORS cmake_find_package
  REQUIRES
    openssl/1.1.1l
    ragel/6.10
    swig/4.0.2
    yasm/1.3.0
    zlib/1.2.12
  IMPORTS
    "bin, *swig* -> ./bin"
    "bin, *yasm* -> ./bin"
    "bin, ragel* -> ./bin"
    "bin, ytasm* -> ./bin"
)
conan_cmake_autodetect(settings)
conan_cmake_install(
  PATH_OR_REFERENCE .
  BUILD missing
  REMOTE conancenter
  SETTINGS ${settings}
)

if (UNIX)
  include(CMakeLists.linux.txt)
elseif (APPLE)
  include(CMakeLists.darwin.txt)
endif()
