# ---------------------------------------------------------------------------
# Build the same .cu sources with the HIP toolchain. Enabled with
# -DHAVE_HIP=ON (which also sets USE_HIP). Only .cu translation units see HIP;
# host C++ is untouched and the CUDA path below is byte-for-byte unchanged. The
# arch comes from CMAKE_HIP_ARCHITECTURES; set -DCMAKE_HIP_ARCHITECTURES=<arch>
# to build a specific architecture with no source edit, otherwise the host GPU
# is auto-detected.
# ---------------------------------------------------------------------------
if (HAVE_HIP)
  if(${CMAKE_VERSION} VERSION_LESS "3.21.0")
    message(FATAL_ERROR "Build with HIP requires at least cmake 3.21.0")
  endif()

  # enable_language(HIP) honors -DCMAKE_HIP_ARCHITECTURES, otherwise auto-detects
  # the host GPU(s) and errors if none is found (no-GPU build host sets the arch).
  enable_language(HIP)

  include(global_flags)
  include(common)

  set(CB_HIP_COMPAT_HEADER ${PROJECT_SOURCE_DIR}/library/cpp/cuda/wrappers/cuda_to_hip.h)
  set(CB_HIP_COMPAT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/library/cpp/cuda/wrappers/hip_compat)

  # Force-include the compat header on every HIP TU so its cuda*->hip* aliases
  # and CB_FULL_WARP_MASK precede any use regardless of per-file include order.
  # clang-cl (the MSVC driver) ignores gcc-style -include
  # (the header then becomes a stray source -> "/Fo with multiple source files");
  # it needs MSVC /FI. The Linux clang gcc-driver uses -include.
  if (CMAKE_HIP_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    string(APPEND CMAKE_HIP_FLAGS " -DUSE_HIP /FI${CB_HIP_COMPAT_HEADER}")
    # On Windows, catboost's hygiene defines live in
    # CMAKE_CXX_FLAGS (global_flags.compiler.msvc) and are not seen by HIP TUs.
    # Without WIN32_LEAN_AND_MEAN, <windows.h> drags in winsock.h which then
    # collides with winsock2.h (redefinition of sockaddr/fd_set/...). NOMINMAX
    # avoids min/max macro clashes in the GPU host code.
    string(APPEND CMAKE_HIP_FLAGS " -DWIN32_LEAN_AND_MEAN -DNOMINMAX -D_WIN32_WINNT=0x0601 -D_CRT_SECURE_NO_WARNINGS -D_USE_MATH_DEFINES")
    # On Windows, catboost ships its own libc++ (libcxxmsvc) on
    # the -I path. TUs that also pull MSVC's STL (e.g. via the unittest framework
    # or ONNX) then see two std::memory_order etc. -> "ambiguous". Exclude MSVC's
    # C++ stdlib so only libcxxmsvc is used (the C UCRT headers are unaffected).
    string(APPEND CMAKE_HIP_FLAGS " /clang:-nostdinc++")
  else()
    string(APPEND CMAKE_HIP_FLAGS " -DUSE_HIP -include ${CB_HIP_COMPAT_HEADER}")
  endif()
  # rocThrust/hipCUB require C++17+; the project is C++20 already, keep it explicit.
  string(APPEND CMAKE_HIP_FLAGS " -std=c++20")

  function(target_cuda_flags Tgt)
    # nvcc-only flags (e.g. -Wno-deprecated-gpu-targets) have no hipcc analogue.
  endfunction()

  function(target_cuda_cflags Tgt)
    # host --compiler-options are passed directly to clang/hipcc; nothing to do.
  endfunction()

  function(target_cuda_sources Tgt Scope)
    # The <cub/...> and <cooperative_groups.h> shim headers must win over the
    # toolkit names; add the shim dir to this target's HIP include path only.
    target_include_directories(${Tgt} PRIVATE ${CB_HIP_COMPAT_INCLUDE_DIR})
    set_source_files_properties(${ARGN} PROPERTIES LANGUAGE HIP)
    set_target_properties(${Tgt} PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
    target_sources(${Tgt} ${Scope} ${ARGN})
  endfunction()

  # The generated CMakeLists hardcode CUDA-toolkit static libs in
  # target_link_options (-lcudart_static -lcudadevrt -lculibos), which do not
  # exist for a HIP build. Filter them out and link the HIP runtime instead, by
  # overriding target_link_options here so the 80+ generated per-target files
  # need no edits.
  set(CB_HIP_RUNTIME_LIBS hip::host hip::device)
  function(target_link_options Tgt)
    set(_filtered "")
    foreach(_opt ${ARGN})
      if (_opt STREQUAL "-lcudart_static" OR _opt STREQUAL "-lcudadevrt" OR _opt STREQUAL "-lculibos")
        continue()
      endif()
      list(APPEND _filtered ${_opt})
    endforeach()
    _target_link_options(${Tgt} ${_filtered})
    if (NOT TARGET ${Tgt})
      return()
    endif()
    get_target_property(_t ${Tgt} TYPE)
    if (_t STREQUAL "EXECUTABLE" OR _t STREQUAL "SHARED_LIBRARY" OR _t STREQUAL "MODULE_LIBRARY")
      target_link_libraries(${Tgt} PRIVATE ${CB_HIP_RUNTIME_LIBS})
    endif()
  endfunction()

  find_package(hip REQUIRED)

  # The generated platform shim does find_package(CUDAToolkit REQUIRED) and links
  # CUDA::toolkit. Provide that target as the HIP runtime and make the CUDAToolkit
  # find_package a no-op so the shim is unchanged (it is the single chokepoint all
  # GPU targets link through).
  if (NOT TARGET CUDA::toolkit)
    add_library(CUDA::toolkit INTERFACE IMPORTED)
    target_link_libraries(CUDA::toolkit INTERFACE hip::host hip::device)
  endif()

  macro(find_package)
    if ("${ARGV0}" STREQUAL "CUDAToolkit")
      # already satisfied by the CUDA::toolkit alias above
    else()
      _find_package(${ARGV})
    endif()
  endmacro()

  return()
endif()

if (HAVE_CUDA)
  if(${CMAKE_VERSION} VERSION_LESS "3.17.0")
      message(FATAL_ERROR "Build with CUDA requires at least cmake 3.17.0")
  endif()

  if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "35-virtual;50-virtual;60-virtual;61-real;70-virtual;75-real;80-real;86-real;89-real;90")
  endif()

  enable_language(CUDA)

  include(global_flags)
  include(common)

  function(quote_if_contains_spaces OutVar Var)
    if (Var MATCHES ".*[ ].*")
      set(${OutVar} "\"${Var}\"" PARENT_SCOPE)
    else()
      set(${OutVar} ${Var} PARENT_SCOPE)
    endif()
  endfunction()

  function(get_cuda_flags_from_cxx_flags OutCudaFlags CxxFlags)
    # OutCudaFlags is an output string
    # CxxFlags is a string

    set(skipList
      -gline-tables-only
      # clang coverage
      -fprofile-instr-generate
      -fcoverage-mapping
      /Zc:inline # disable unreferenced functions (kernel registrators) remove
      -Wno-c++17-extensions
      -flto
      -faligned-allocation
      -fsized-deallocation
      # While it might be reasonable to compile host part of .cu sources with these optimizations enabled,
      # nvcc passes these options down towards cicc which lacks x86_64 extensions support.
      -msse2
      -msse3
      -mssse3
      -msse4.1
      -msse4.2
    )

    set(skipPrefixRegexp
      "(-fsanitize=|-fsanitize-coverage=|-fsanitize-blacklist=|--system-header-prefix|(/|-)std(:|=)c\\+\\+).*"
    )

    string(FIND "${CMAKE_CUDA_HOST_COMPILER}" clang hostCompilerIsClangPos)
    string(COMPARE NOTEQUAL ${hostCompilerIsClangPos} -1 isHostCompilerClang)


    function(separate_arguments_with_special_symbols Output Src)
      string(REPLACE ";" "$<SEMICOLON>" LocalOutput "${Src}")
      separate_arguments(LocalOutput NATIVE_COMMAND ${LocalOutput})
      set(${Output} ${LocalOutput} PARENT_SCOPE)
    endfunction()

    separate_arguments_with_special_symbols(Separated_CxxFlags "${CxxFlags}")

    if (MSVC)
      set(flagPrefixSymbol "/")
    else()
      set(flagPrefixSymbol "-")
    endif()

    set(localCudaCommonFlags "") # non host compiler options
    set(localCudaCompilerOptions "")

    while (Separated_CxxFlags)
      list(POP_FRONT Separated_CxxFlags cxxFlag)
      if ((cxxFlag IN_LIST skipList) OR (cxxFlag MATCHES ${skipPrefixRegexp}))
        continue()
      endif()
      if ((cxxFlag STREQUAL -fopenmp=libomp) AND (NOT isHostCompilerClang))
        list(APPEND localCudaCompilerOptions -fopenmp)
        continue()
      endif()
      if ((NOT isHostCompilerClang) AND (cxxFlag MATCHES "^\-\-target=.*"))
        continue()
      endif()
      if (cxxFlag MATCHES "^${flagPrefixSymbol}(D[^ ]+)=(.+)")
        set(key ${CMAKE_MATCH_1})
        quote_if_contains_spaces(safeValue "${CMAKE_MATCH_2}")
        list(APPEND localCudaCommonFlags "-${key}=${safeValue}")
        continue()
      endif()
      if (cxxFlag MATCHES "^${flagPrefixSymbol}([DI])(.*)")
        set(key ${CMAKE_MATCH_1})
        if (CMAKE_MATCH_2)
          set(value ${CMAKE_MATCH_2})
          set(sep "")
        else()
          list(POP_FRONT Separated_CxxFlags value)
          set(sep " ")
        endif()
        quote_if_contains_spaces(safeValue "${value}")
        list(APPEND localCudaCommonFlags "-${key}${sep}${safeValue}")
        continue()
      endif()
      list(APPEND localCudaCompilerOptions ${cxxFlag})
    endwhile()

    if (isHostCompilerClang)
      # nvcc concatenates the sources for clang, and clang reports unused
      # things from .h files as if they they were defined in a .cpp file.
      list(APPEND localCudaCommonFlags -Wno-unused-function -Wno-unused-parameter)
      if (CMAKE_CXX_COMPILER_TARGET)
        list(APPEND localCudaCompilerOptions "--target=${CMAKE_CXX_COMPILER_TARGET}")
      endif()
    endif()

    if (CMAKE_SYSROOT)
      list(APPEND localCudaCompilerOptions "--sysroot=${CMAKE_SYSROOT}")
    endif()

    list(JOIN localCudaCommonFlags " " joinedLocalCudaCommonFlags)
    string(REPLACE "$<SEMICOLON>" ";" joinedLocalCudaCommonFlags "${joinedLocalCudaCommonFlags}")
    list(JOIN localCudaCompilerOptions , joinedLocalCudaCompilerOptions)
    set(${OutCudaFlags} "${joinedLocalCudaCommonFlags} --compiler-options ${joinedLocalCudaCompilerOptions}" PARENT_SCOPE)
  endfunction()

  get_cuda_flags_from_cxx_flags(CMAKE_CUDA_FLAGS "${CMAKE_CXX_FLAGS}")

  string(APPEND CMAKE_CUDA_FLAGS
    # Allow __host__, __device__ annotations in lambda declaration.
    " --expt-extended-lambda"
    # Allow host code to invoke __device__ constexpr functions and vice versa
    " --expt-relaxed-constexpr"
    # Allow to use newer compilers than CUDA Toolkit officially supports
    " --allow-unsupported-compiler"
    # Allow to use libc++ with CUDA 12.3+
    " -D_ALLOW_UNSUPPORTED_LIBCPP"
  )

  set(NVCC_STD_VER 17)
  if(MSVC)
    set(NVCC_STD "/std:c++${NVCC_STD_VER}")
  else()
    set(NVCC_STD "-std=c++${NVCC_STD_VER}")
  endif()
  string(APPEND CMAKE_CUDA_FLAGS " --compiler-options ${NVCC_STD}")

  string(APPEND CMAKE_CUDA_FLAGS " -DTHRUST_IGNORE_CUB_VERSION_CHECK")

  if(MSVC)
    # default CMake flags differ from our configuration
    set(CMAKE_CUDA_FLAGS_DEBUG "-D_DEBUG --compiler-options /Z7,/Ob0,/Od")
    set(CMAKE_CUDA_FLAGS_MINSIZEREL "-DNDEBUG --compiler-options /O1,/Ob1")
    set(CMAKE_CUDA_FLAGS_RELEASE "-DNDEBUG --compiler-options /Ox,/Ob2,/Oi")
    set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-DNDEBUG --compiler-options /Z7,/Ox,/Ob1")
  endif()

  # use versions from contrib, standard libraries from CUDA distibution are incompatible with MSVC and libcxx
  set(CUDA_EXTRA_INCLUDE_DIRECTORIES
    ${PROJECT_SOURCE_DIR}/contrib/deprecated/nvidia/thrust
    ${PROJECT_SOURCE_DIR}/contrib/deprecated/nvidia/cub
  )

  find_package(CUDAToolkit REQUIRED)

  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "11.2")
    string(APPEND CMAKE_CUDA_FLAGS " --threads 2")
  endif()

  message(VERBOSE "CMAKE_CUDA_FLAGS = \"${CMAKE_CUDA_FLAGS}\"")

  enable_language(CUDA)

  function(target_cuda_flags Tgt)
    set_property(TARGET ${Tgt} APPEND PROPERTY
      CUDA_FLAGS ${ARGN}
    )
  endfunction()

  function(target_cuda_cflags Tgt)
    if (NOT ("${ARGN}" STREQUAL ""))
      string(JOIN "," OPTIONS ${ARGN})
      set_property(TARGET ${Tgt} APPEND PROPERTY
        CUDA_FLAGS --compiler-options ${OPTIONS}
      )
    endif()
  endfunction()

  function(target_cuda_sources Tgt Scope)
    # add include directories on per-CMakeLists file level because some non-CUDA source files may want to include calls to CUDA libs
    include_directories(${CUDA_EXTRA_INCLUDE_DIRECTORIES})

    set_source_files_properties(${ARGN} PROPERTIES
      COMPILE_OPTIONS "$<JOIN:$<TARGET_GENEX_EVAL:${Tgt},$<TARGET_PROPERTY:${Tgt},CUDA_FLAGS>>,;>"
    )
    target_sources(${Tgt} ${Scope} ${ARGN})
  endfunction()

endif()
