if(${CMAKE_CXX_COMPILER_ID} STREQUAL Clang
    AND "${CMAKE_CXX_COMPILER_FRONTEND_VARIANT}" STREQUAL "MSVC"
    AND "${CMAKE_CXX_SIMULATE_ID}" STREQUAL "MSVC")

  set(_IS_CLANG_CL_COMPILER true)
else()
  set(_IS_CLANG_CL_COMPILER false)
endif()

set(_WARNS_ENABLED
  4018  # 'expression' : signed/unsigned mismatch
  4265  # 'class' : class has virtual functions, but destructor is not virtual
  4296  # 'operator' : expression is always false
  4431  # missing type specifier - int assumed
)

set(_WARNS_AS_ERROR
  4013  # 'function' undefined; assuming extern returning int
)

set(_WARNS_DISABLED
    # While this warning corresponds to enabled-by-default -Wmacro-redefinition,
    # it floods clog with abundant amount of log lines,
    # as yvals_core.h from Windows SDK redefines certain
    # which macros logically belong to libcxx
    4005  # '__cpp_lib_*': macro redefinition.

    # Ne need to recheck this, but it looks like _CRT_USE_BUILTIN_OFFSETOF still makes sense
    4117  # macro name '_CRT_USE_BUILTIN_OFFSETOF' is reserved, '#define' ignored

    4127  # conditional expression is constant
    4200  # nonstandard extension used : zero-sized array in struct/union
    4201  # nonstandard extension used : nameless struct/union
    4351  # elements of array will be default initialized
    4355  # 'this' : used in base member initializer list
    4503  # decorated name length exceeded, name was truncated
    4510  # default constructor could not be generated
    4511  # copy constructor could not be generated
    4512  # assignment operator could not be generated
    4554  # check operator precedence for possible error; use parentheses to clarify precedence
    4610  # 'object' can never be instantiated - user defined constructor required
    4706  # assignment within conditional expression
    4800  # forcing value to bool 'true' or 'false' (performance warning)
    4996  # The POSIX name for this item is deprecated
    4714  # function marked as __forceinline not inlined
    4197  # 'TAtomic' : top-level volatile in cast is ignored
    4245  # 'initializing' : conversion from 'int' to 'ui32', signed/unsigned mismatch
    4324  # 'ystd::function<void (uint8_t *)>': structure was padded due to alignment specifier
    5033  # 'register' is no longer a supported storage class
)

set (_MSVC_COMMON_C_CXX_FLAGS " \
  /DWIN32 \
  /D_WIN32 \
  /D_WINDOWS \
  /D_CRT_SECURE_NO_WARNINGS \
  /D_CRT_NONSTDC_NO_WARNINGS \
  /D_USE_MATH_DEFINES \
  /D__STDC_CONSTANT_MACROS \
  /D__STDC_FORMAT_MACROS \
  /D_USING_V110_SDK71_ \
  /DWIN32_LEAN_AND_MEAN \
  /DNOMINMAX \
  /nologo \
  /Zm500 \
  /GR \
  /bigobj \
  /FC \
  /EHs \
  /errorReport:prompt \
  /Zc:inline \
  /utf-8 \
  /permissive- \
  /D_WIN32_WINNT=0x0601 \
  /D_MBCS \
")

if (NOT _IS_CLANG_CL_COMPILER)
  # unused by clang-cl
  string(APPEND _MSVC_COMMON_C_CXX_FLAGS " /MP")
endif()

if (CMAKE_GENERATOR MATCHES "Visual.Studio.*")
  string(APPEND _MSVC_COMMON_C_CXX_FLAGS "\
    /DY_UCRT_INCLUDE=\"$(UniversalCRT_IncludePath.Split(';')[0].Replace('\\','/'))\" \
    /DY_MSVC_INCLUDE=\"$(VC_VC_IncludePath.Split(';')[0].Replace('\\','/'))\" \
  ")
else()
  set(UCRT_INCLUDE_FOUND false)
  foreach(INCLUDE_PATH $ENV{INCLUDE})
    if (INCLUDE_PATH MATCHES ".*\\\\Windows Kits\\\\[0-9]+\\\\include\\\\[0-9\\.]+\\\\ucrt$")
      message(VERBOSE "Found Y_UCRT_INCLUDE path \"${INCLUDE_PATH}\"")
      string(REPLACE "\\" "/" SAFE_INCLUDE_PATH "${INCLUDE_PATH}")
      string(APPEND _MSVC_COMMON_C_CXX_FLAGS " /DY_UCRT_INCLUDE=\"${SAFE_INCLUDE_PATH}\"")
      set(UCRT_INCLUDE_FOUND true)
      break()
    endif()
  endforeach()
  if (NOT UCRT_INCLUDE_FOUND)
    message(FATAL_ERROR "UniversalCRT include path not found, please add it to the standard INCLUDE environment variable (most likely by calling vcvars64.bat)")
  endif()

  set(MSVC_INCLUDE_FOUND false)
  foreach(INCLUDE_PATH $ENV{INCLUDE})
    if (INCLUDE_PATH MATCHES ".*VC\\\\Tools\\\\MSVC\\\\[0-9\\.]+\\\\include$")
      message(VERBOSE "Found Y_MSVC_INCLUDE path \"${INCLUDE_PATH}\"")
      string(REPLACE "\\" "/" SAFE_INCLUDE_PATH "${INCLUDE_PATH}")
      string(APPEND _MSVC_COMMON_C_CXX_FLAGS " /DY_MSVC_INCLUDE=\"${SAFE_INCLUDE_PATH}\"")
      set(MSVC_INCLUDE_FOUND true)
      break()
    endif()
  endforeach()
  if (NOT MSVC_INCLUDE_FOUND)
    message(FATAL_ERROR "MSVC include path not found, please add it to the standard INCLUDE environment variable (most likely by calling vcvars64.bat)")
  endif()
endif()

foreach(WARN ${_WARNS_AS_ERROR})
  string(APPEND _MSVC_COMMON_C_CXX_FLAGS " /we${WARN}")
endforeach()

foreach(WARN ${_WARNS_ENABLED})
  string(APPEND _MSVC_COMMON_C_CXX_FLAGS " /w1${WARN}")
endforeach()

foreach(WARN ${_WARNS_DISABLED})
  string(APPEND _MSVC_COMMON_C_CXX_FLAGS " /wd${WARN}")
endforeach()

if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
  string(APPEND _MSVC_COMMON_C_CXX_FLAGS " \
    /D_WIN64 \
    /DWIN64 \
    /D__SSE2__ \
    /D__SSE3__ \
    /D__SSSE3__ \
    /D__SSE4_1__ \
    /D__SSE4_2__ \
    /D__POPCNT__ \
  ")
endif()

if (_IS_CLANG_CL_COMPILER)
  # clang-cl works slighly differently than MSVC compiler when specifying arch options, so we have to set them differently
  # https://github.com/llvm/llvm-project/issues/56722

  include(global_flags.compiler.gnu.march)
  string(APPEND _MSVC_COMMON_C_CXX_FLAGS " ${_GNU_MARCH_C_CXX_FLAGS}")
endif()


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${_MSVC_COMMON_C_CXX_FLAGS} \
")

# TODO - '/D_CRT_USE_BUILTIN_OFFSETOF'
# TODO - -DUSE_STL_SYSTEM

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_MSVC_COMMON_C_CXX_FLAGS} \
  /std:c++latest \
  /Zc:__cplusplus \
")

set(CMAKE_CXX_FLAGS_DEBUG "/Z7 /Ob0 /Od /D_DEBUG")
set(CMAKE_CXX_FLAGS_MINSIZEREL "/O1 /Ob1 /DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "/Ox /Ob2 /Oi /DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/Z7 /Ox /Ob1 /DNDEBUG")
