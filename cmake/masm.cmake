if (MSVC)

enable_language(ASM_MASM)

macro(curdir_masm_flags)
  set(CMAKE_ASMMASM_FLAGS ${ARGN})
endmacro()

endif()
