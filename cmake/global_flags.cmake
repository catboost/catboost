set(CMAKE_C_FLAGS "")
set(CMAKE_CXX_FLAGS "")

# workaround when old NDK toolchain that does not set CMAKE_ANDROID_NDK_VERSION is used
# See for details: https://gitlab.kitware.com/cmake/cmake/-/issues/24386
if(ANDROID_NDK_REVISION AND NOT CMAKE_ANDROID_NDK_VERSION)
  set(CMAKE_ANDROID_NDK_VERSION "${ANDROID_NDK_REVISION}")
endif()

if (MSVC)
  set(flagPrefixSymbol "/")
  include(global_flags.compiler.msvc)
  include(global_flags.linker.msvc)
else()
  set(flagPrefixSymbol "-")
  include(global_flags.compiler.gnu)
  include(global_flags.linker.gnu)
endif()

if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(i686|x86_64|AMD64)$")
  set(_ALL_X86_EXTENSIONS_DEFINES "\
    ${flagPrefixSymbol}DSSE_ENABLED=1 \
    ${flagPrefixSymbol}DSSE3_ENABLED=1 \
    ${flagPrefixSymbol}DSSSE3_ENABLED=1 \
  ")
  if ((CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$") OR (NOT ANDROID))
    string(APPEND _ALL_X86_EXTENSIONS_DEFINES "\
      ${flagPrefixSymbol}DSSE41_ENABLED=1 \
      ${flagPrefixSymbol}DSSE42_ENABLED=1 \
      ${flagPrefixSymbol}DPOPCNT_ENABLED=1 \
    ")
    if (NOT ANDROID)
      # older clang versions did not support this feature on Android:
      # https://reviews.llvm.org/rGc32d307a49f5255602e7543e64e6c38a7f536abc
      string(APPEND _ALL_X86_EXTENSIONS_DEFINES " ${flagPrefixSymbol}DCX16_ENABLED=1")
    endif()
  endif()

  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${_ALL_X86_EXTENSIONS_DEFINES}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_ALL_X86_EXTENSIONS_DEFINES}")
endif()

message(VERBOSE "CMAKE_C_FLAGS = \"${CMAKE_C_FLAGS}\"")
message(VERBOSE "CMAKE_CXX_FLAGS = \"${CMAKE_CXX_FLAGS}\"")

if (NOT CMAKE_CROSSCOMPILING)
  set(TOOLS_ROOT ${PROJECT_BINARY_DIR})
elseif(NOT TOOLS_ROOT)
  message(FATAL_ERROR "TOOLS_ROOT is required for crosscompilation")
endif()
