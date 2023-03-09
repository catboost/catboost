function(target_cython_include_directories Tgt)
  set_property(TARGET ${Tgt} APPEND PROPERTY
    CYTHON_INCLUDE_DIRS ${ARGN}
  )
endfunction()

function(target_cython_options Tgt)
  set_property(TARGET ${Tgt} APPEND PROPERTY
    CYTHON_OPTIONS ${ARGN}
  )
endfunction()

macro(set_python_type_for_cython Type)
  if (${Type} STREQUAL PY3)
    find_package(Python3 REQUIRED)
    set(PYTHON Python3::Interpreter)
  else()
    find_package(Python2 REQUIRED)
    set(PYTHON Python2::Interpreter)
  endif()
endmacro()

function(target_cython_sources Tgt Scope)
  foreach(Input ${ARGN})
    get_filename_component(OutputBase ${Input} NAME)
    set(CppCythonOutput ${CMAKE_CURRENT_BINARY_DIR}/${OutputBase}.cpp)
    add_custom_command(
      OUTPUT ${CppCythonOutput}
      COMMAND ${PYTHON} ${CMAKE_SOURCE_DIR}/contrib/tools/cython/cython.py ${Input} -o ${CppCythonOutput}
        "$<JOIN:$<TARGET_GENEX_EVAL:${Tgt},$<TARGET_PROPERTY:${Tgt},CYTHON_OPTIONS>>,$<SEMICOLON>>"
        "-I$<JOIN:$<TARGET_GENEX_EVAL:${Tgt},$<TARGET_PROPERTY:${Tgt},CYTHON_INCLUDE_DIRS>>,$<SEMICOLON>-I>"
      COMMAND_EXPAND_LISTS
      DEPENDS ${OUTPUT_INCLUDES}
    )
    target_sources(${Tgt} ${Scope} ${CppCythonOutput})
  endforeach()
endfunction()
