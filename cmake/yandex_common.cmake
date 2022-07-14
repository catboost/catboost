# Set of common macros

find_package(Python2 REQUIRED)
find_package(Python3 REQUIRED)

add_compile_definitions(CATBOOST_OPENSOURCE=yes)

function(target_ragel_lexers TgtName Key Src)
  SET(RAGEL_BIN ${CMAKE_BINARY_DIR}/bin/ragel)
  get_filename_component(OutPath ${Src} NAME)
  string(APPEND OutPath .cpp)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${OutPath}
    COMMAND Python2::Interpreter ${CMAKE_SOURCE_DIR}/build/scripts/run_tool.py -- ${RAGEL_BIN} ${RAGEL_FLAGS} ${ARGN} -o ${CMAKE_CURRENT_BINARY_DIR}/${OutPath} ${Src}
    DEPENDS ${CMAKE_SOURCE_DIR}/build/scripts/run_tool.py ${Src}
  )
  target_sources(${TgtName} ${Key} ${CMAKE_CURRENT_BINARY_DIR}/${OutPath})
endfunction()

function(target_yasm_source TgtName Key Src)
  SET(YASM_BIN ${CMAKE_BINARY_DIR}/bin/yasm)
  get_filename_component(OutPath ${Src} NAME_WLE)
  string(APPEND OutPath .o)
  add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${OutPath}
      COMMAND Python2::Interpreter ${CMAKE_SOURCE_DIR}/build/scripts/run_tool.py -- ${YASM_BIN} ${YASM_FLAGS} ${ARGN} -o ${CMAKE_CURRENT_BINARY_DIR}/${OutPath} ${Src}
    DEPENDS ${CMAKE_SOURCE_DIR}/build/scripts/run_tool.py ${Src}
  )
  target_sources(${TgtName} ${Key} ${CMAKE_CURRENT_BINARY_DIR}/${OutPath})
endfunction()

function(target_joined_source TgtName Out)
  foreach(InSrc ${ARGN})
    file(RELATIVE_PATH IncludePath ${CMAKE_SOURCE_DIR} ${InSrc})
    list(APPEND IncludesList ${IncludePath})
  endforeach()
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${Out}
    COMMAND Python2::Interpreter ${CMAKE_SOURCE_DIR}/build/scripts/gen_join_srcs.py ${CMAKE_CURRENT_BINARY_DIR}/${Out} ${IncludesList}
    DEPENDS ${CMAKE_SOURCE_DIR}/build/scripts/gen_join_srcs.py ${ARGN}
  )
  target_sources(${TgtName} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/${Out})
endfunction()

function(generate_enum_serilization Tgt Input)
  set(opts "")
  set(oneval_args INCLUDE_HEADERS)
  set(multival_args "")
  cmake_parse_arguments(ENUM_SERIALIZATION_ARGS
    "${opts}"
    "${oneval_args}"
    "${multival_args}"
    ${ARGN}
  )
  get_filename_component(BaseName ${Input} NAME)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${BaseName}_serialized.cpp
    COMMAND enum_parser ${Input} --include-path ${ENUM_SERIALIZATION_ARGS_INCLUDE_HEADERS} --output ${CMAKE_CURRENT_BINARY_DIR}/${BaseName}_serialized.cpp
    DEPENDS ${Input}
  )
  target_sources(${Tgt} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/${BaseName}_serialized.cpp)
endfunction()

function(add_global_library_for TgtName MainName)
  add_library(${TgtName} STATIC ${ARGN})
  add_library(${TgtName}.wholearchive INTERFACE)
  add_dependencies(${TgtName}.wholearchive ${TgtName})
  add_dependencies(${TgtName} ${MainName})
  if (MSVC)
    target_link_options(${TgtName}.wholearchive INTERFACE "SHELL:/WHOLEARCHIVE:$<TARGET_FILE:${TgtName}>")
  elseif(APPLE)
    target_link_options(${TgtName}.wholearchive INTERFACE "SHELL:-Wl,-force_load,$<TARGET_FILE:${TgtName}>")
  else()
    target_link_options(${TgtName}.wholearchive INTERFACE "SHELL:-Wl,--whole-archive $<TARGET_FILE:${TgtName}> -Wl,--no-whole-archive")
  endif()
  target_link_libraries(${MainName} INTERFACE ${TgtName}.wholearchive)
endfunction()

function(target_link_flags)
  target_link_libraries(${ARGN})
endfunction()

function(copy_file From To)
  add_custom_command(
    OUTPUT ${To}
    COMMAND ${CMAKE_COMMAND} -E copy ${From} ${To}
    DEPENDS ${From}
  )
endfunction()

function(vcs_info Tgt)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/__vcs_version__.c
    COMMAND Python2::Interpreter ${CMAKE_SOURCE_DIR}/build/scripts/vcs_info.py no-vcs dummy.json ${CMAKE_CURRENT_BINARY_DIR}/__vcs_version__.c ${CMAKE_SOURCE_DIR}/build/scripts/c_templates/svn_interface.c
    DEPENDS ${CMAKE_SOURCE_DIR}/build/scripts/vcs_info.py ${CMAKE_SOURCE_DIR}/build/scripts/c_templates/svn_interface.c
  )
  target_sources(${Tgt} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/__vcs_version__.c)
endfunction()

function(resources Tgt Output)
  set(opts "")
  set(oneval_args "")
  set(multival_args INPUTS KEYS)
  cmake_parse_arguments(RESOURCE_ARGS
    "${opts}"
    "${oneval_args}"
    "${multival_args}"
    ${ARGN}
  )
  list(LENGTH RESOURCE_ARGS_INPUTS InputsCount)
  list(LENGTH RESOURCE_ARGS_KEYS KeysCount)
  if (NOT ${InputsCount} EQUAL ${KeysCount})
    message(FATAL_ERROR "Resources inputs count isn't equal to keys count in " ${Tgt})
  endif()
  math(EXPR ListsMaxIdx "${InputsCount} - 1")
  foreach(Idx RANGE ${ListsMaxIdx})
    list(GET RESOURCE_ARGS_INPUTS ${Idx} Input)
    list(GET RESOURCE_ARGS_KEYS ${Idx} Key)
    list(APPEND ResourcesList ${Input})
    list(APPEND ResourcesList ${Key})
  endforeach()
  add_custom_command(
    OUTPUT ${Output}
    COMMAND rescompiler ${Output} ${ResourcesList}
    DEPENDS ${RESOURCE_ARGS_INPUTS}
  )
endfunction()

function(use_export_script Target ExportFile)
  get_filename_component(OutName ${ExportFile} NAME)
  set(OutPath ${CMAKE_CURRENT_BINARY_DIR}/gen_${OutName})

  if (MSVC)
    target_link_flags(${Target} PRIVATE /DEF:${OutPath})
    set(EXPORT_SCRIPT_FLAVOR msvc)
  elseif(APPLE)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/build/scripts/export_script_gen.py ${ExportFile} - --format darwin
      RESULT_VARIABLE _SCRIPT_RES
      OUTPUT_VARIABLE _SCRIPT_FLAGS
      ERROR_VARIABLE _SCRIPT_STDERR
    )
    if (NOT ${_SCRIPT_RES} EQUAL 0)
      message(FATAL_ERROR "Failed to parse export symbols from ${ExportFile}:\n${_SCRIPT_STDERR}")
      return()
    endif()
    target_link_flags(${Target} PRIVATE ${_SCRIPT_FLAGS})
    return()
  else()
    set(EXPORT_SCRIPT_FLAVOR gnu)
    target_link_flags(${Target} PRIVATE -Wl,--gc-sections -rdynamic -Wl,--version-script=${OutPath})
  endif()

  add_custom_command(
    OUTPUT ${OutPath}
    COMMAND
      Python3::Interpreter ${CMAKE_SOURCE_DIR}/build/scripts/export_script_gen.py ${ExportFile} ${OutPath} --format ${EXPORT_SCRIPT_FLAVOR}
    DEPENDS ${ExportFile} ${CMAKE_SOURCE_DIR}/build/scripts/export_script_gen.py
  )
  target_sources(${Target} PRIVATE ${OutPath})
  set_property(SOURCE ${OutPath} PROPERTY
    HEADER_FILE_ONLY On
  )
  set_property(TARGET ${Target} APPEND PROPERTY
    LINK_DEPENDS ${OutPath}
  )
endfunction()
