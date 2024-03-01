if (HAVE_CUDA)
  if(${CMAKE_VERSION} VERSION_LESS "3.17.0")
      message(FATAL_ERROR "Build with CUDA requires at least cmake 3.17.0")
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
    ${PROJECT_SOURCE_DIR}/contrib/libs/nvidia/thrust
    ${PROJECT_SOURCE_DIR}/contrib/libs/nvidia/cub
  )

  find_package(CUDAToolkit REQUIRED)

  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "11.2")
    string(APPEND CMAKE_CUDA_FLAGS " --threads 0")
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
