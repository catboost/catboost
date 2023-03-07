set(CMAKE_C_FLAGS "")
set(CMAKE_CXX_FLAGS "")

if (MSVC)
  include(global_flags.compiler.msvc)
  include(global_flags.linker.msvc)
else()
  include(global_flags.compiler.gnu)
  include(global_flags.linker.gnu)
endif()

if ((CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64") OR (CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64"))
  set(_X86_64_DEFINES "\
    -DSSE_ENABLED=1 \
    -DSSE3_ENABLED=1 \
    -DSSSE3_ENABLED=1 \
    -DSSE41_ENABLED=1 \
    -DSSE42_ENABLED=1 \
    -DPOPCNT_ENABLED=1 \
    -DCX16_ENABLED=1 \
  ")

  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${_X86_64_DEFINES}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_X86_64_DEFINES}")
endif()

if (NOT CMAKE_CROSSCOMPILING)
  set(TOOLS_ROOT ${CMAKE_BINARY_DIR})
elseif(NOT TOOLS_ROOT)
  message(FATAL_ERROR "TOOLS_ROOT is required for crosscompilation")
endif()
