if (MSVC)

enable_language(MASM)

macro(curdir_masm_flags)
  set(CMAKE_ASMMASM_FLAGS ${ARGN})
endmacro()

endif()
