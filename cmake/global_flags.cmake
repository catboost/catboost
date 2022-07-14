set(CMAKE_C_FLAGS "\
  -m64 \
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
if (APPLE)
  set(CMAKE_SHARED_LINKER_FLAGS "-undefined dynamic_lookup")
elseif(UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fuse-init-array")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fuse-init-array")
endif()
