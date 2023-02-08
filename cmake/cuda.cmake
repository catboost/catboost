if (NOT APPLE AND NOT ANDROID AND NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
  if(${CMAKE_VERSION} VERSION_LESS "3.17.0") 
      message(FATAL_ERROR "Build with CUDA requires at least cmake 3.17.0")
  endif()

  enable_language(CUDA)

  set(CUDA_DEFINES
    -D_THREAD_SAFE
    -D_PTHREADS
    -D_REENTRANT
    -D_LARGEFILE_SOURCE
    -D__STDC_CONSTANT_MACROS
    -D__STDC_FORMAT_MACROS
    -D_FILE_OFFSET_BITS=64
    -D_GNU_SOURCE
    -D_YNDX_LIBUNWIND_ENABLE_EXCEPTION_BACKTRACE
    -D__LONG_LONG_SUPPORTED
    -DSSE_ENABLED=1
    -DSSE3_ENABLED=1
    -DSSSE3_ENABLED=1
    -DSSE41_ENABLED=1
    -DSSE42_ENABLED=1
    -DPOPCNT_ENABLED=1
    -DCX16_ENABLED=1
    -D_libunwind_
    -DLIBCXX_BUILDING_LIBCXXRT
  )
  set(CUDA_HOST_FLAGS
    -m64
    -fexceptions
    -fno-common
    -fuse-init-array
    -fcolor-diagnostics
    -ffunction-sections
    -fdata-sections
    -Wall
    -Wextra
    -Wno-parentheses
    -Wno-implicit-const-int-float-conversion
    -Wno-unknown-warning-option
    -msse2
    -msse3
    -mssse3
    -msse4.1
    -msse4.2
    -mpopcnt
    -mcx16
    -Woverloaded-virtual
    -Wimport-preprocessor-directive-pedantic
    -Wno-ambiguous-reversed-operator
    -Wno-defaulted-function-deleted
    -Wno-deprecated-anon-enum-enum-conversion
    -Wno-deprecated-enum-enum-conversion
    -Wno-deprecated-enum-float-conversion
    -Wno-deprecated-volatile
    -Wno-pessimizing-move
    -Wno-range-loop-construct
    -Wno-return-std-move
    -Wno-undefined-var-template
    -nostdinc++
    -Wno-unused-function
    -Wno-unused-parameter
  )

  find_package(CUDAToolkit REQUIRED)

  enable_language(CUDA)

  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CLANGPLUSPLUS ${CMAKE_CXX_COMPILER})
    message(STATUS "Using ${CLANGPLUSPLUS} for cuda compilation")
  else()
    find_program(CLANGPLUSPLUS NAMES clang++-12 clang++)
    if (CLANGPLUSPLUS MATCHES "CLANGPLUSPLUS-NOTFOUND")
      message(SEND_ERROR "clang++ not found")
    else()
      message(STATUS "Using ${CLANGPLUSPLUS} for cuda compilation")
    endif()
  endif()

  set(CUDACXX ${CLANGPLUSPLUS})

  link_directories(${CUDAToolkit_LIBRARY_DIR})
  include_directories(SYSTEM ${CUDAToolkit_INCLUDE_DIRS})
  add_compile_definitions(THRUST_IGNORE_CUB_VERSION_CHECK)

  function(target_cuda_flags Tgt)
    set_property(TARGET ${Tgt} APPEND PROPERTY
      CUDA_FLAGS --compiler-bindir=${CLANGPLUSPLUS} ${CUDA_DEFINES} ${ARGN}
    )
  endfunction()

  function(target_cuda_cflags Tgt)
    string(JOIN "," OPTIONS ${CUDA_HOST_FLAGS} ${ARGN})
    set_property(TARGET ${Tgt} APPEND PROPERTY
      CUDA_FLAGS --compiler-options ${OPTIONS}
    )
  endfunction()

  function(target_cuda_sources Tgt Scope)
    set_source_files_properties(${ARGN} PROPERTIES
      COMPILE_OPTIONS "$<JOIN:$<TARGET_GENEX_EVAL:${Tgt},$<TARGET_PROPERTY:${Tgt},CUDA_FLAGS>>,;>"
    )
    target_sources(${Tgt} ${Scope} ${ARGN})
  endfunction()

endif()
