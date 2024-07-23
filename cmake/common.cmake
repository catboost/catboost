# Set of common macros

find_package(Python3 REQUIRED)

add_compile_definitions(ARCADIA_ROOT=${PROJECT_SOURCE_DIR})
add_compile_definitions(ARCADIA_BUILD_ROOT=${PROJECT_BINARY_DIR})
add_compile_definitions(CATBOOST_OPENSOURCE=yes)

# assumes ToolName is always both the binary and the target name
function(get_built_tool_path OutBinPath OutDependency SrcPath ToolName)
  if (CMAKE_GENERATOR MATCHES "Visual.Studio.*")
    set(BinPath "${TOOLS_ROOT}/${SrcPath}/\$(Configuration)/${ToolName}${CMAKE_EXECUTABLE_SUFFIX}")
  else()
    set(BinPath "${TOOLS_ROOT}/${SrcPath}/${ToolName}${CMAKE_EXECUTABLE_SUFFIX}")
  endif()
  set(${OutBinPath} ${BinPath} PARENT_SCOPE)
  if (CMAKE_CROSSCOMPILING)
    set(${OutDependency} ${BinPath} PARENT_SCOPE)
  else()
    set(${OutDependency} ${ToolName} PARENT_SCOPE)
  endif()
endfunction()


function(target_ragel_lexers TgtName Key Src)
  SET(RAGEL_BIN ${PROJECT_BINARY_DIR}/bin/ragel${CMAKE_EXECUTABLE_SUFFIX})
  get_filename_component(OutPath ${Src} NAME_WLE)
  get_filename_component(SrcDirPath ${Src} DIRECTORY)
  get_filename_component(OutputExt ${OutPath} EXT)
  if (OutputExt STREQUAL "")
    string(APPEND OutPath .rl6.cpp)
  endif()
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${OutPath}
    COMMAND Python3::Interpreter ${PROJECT_SOURCE_DIR}/build/scripts/run_tool.py -- ${RAGEL_BIN} ${RAGEL_FLAGS} ${ARGN} -o ${CMAKE_CURRENT_BINARY_DIR}/${OutPath} ${Src}
    DEPENDS ${PROJECT_SOURCE_DIR}/build/scripts/run_tool.py ${Src}
    WORKING_DIRECTORY ${SrcDirPath}
  )
  target_sources(${TgtName} ${Key} ${CMAKE_CURRENT_BINARY_DIR}/${OutPath})
endfunction()

function(target_yasm_source TgtName Key Src)
  SET(YASM_BIN ${PROJECT_BINARY_DIR}/bin/yasm${CMAKE_EXECUTABLE_SUFFIX})
  get_filename_component(OutPath ${Src} NAME_WLE)
  string(APPEND OutPath .o)
  add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${OutPath}
      COMMAND Python3::Interpreter ${PROJECT_SOURCE_DIR}/build/scripts/run_tool.py -- ${YASM_BIN} ${YASM_FLAGS} ${ARGN} -o ${CMAKE_CURRENT_BINARY_DIR}/${OutPath} ${Src}
    DEPENDS ${PROJECT_SOURCE_DIR}/build/scripts/run_tool.py ${Src}
  )
  target_sources(${TgtName} ${Key} ${CMAKE_CURRENT_BINARY_DIR}/${OutPath})
endfunction()

function(target_joined_source TgtName Out)
  foreach(InSrc ${ARGN})
    file(RELATIVE_PATH IncludePath ${PROJECT_SOURCE_DIR} ${InSrc})
    list(APPEND IncludesList ${IncludePath})
  endforeach()
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${Out}
    COMMAND Python3::Interpreter ${PROJECT_SOURCE_DIR}/build/scripts/gen_join_srcs.py ${CMAKE_CURRENT_BINARY_DIR}/${Out} ${IncludesList}
    DEPENDS ${PROJECT_SOURCE_DIR}/build/scripts/gen_join_srcs.py ${ARGN}
  )
  target_sources(${TgtName} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/${Out})
endfunction()

function(target_sources_custom TgtName CompileOutSuffix)
  set(opts "")
  set(oneval_args "")
  set(multival_args SRCS CUSTOM_FLAGS)
  cmake_parse_arguments(TARGET_SOURCES_CUSTOM
    "${opts}"
    "${oneval_args}"
    "${multival_args}"
    ${ARGN}
  )

  foreach(Src ${TARGET_SOURCES_CUSTOM_SRCS})
    file(RELATIVE_PATH SrcRealPath ${PROJECT_SOURCE_DIR} ${Src})
    get_filename_component(SrcDir ${SrcRealPath} DIRECTORY)
    get_filename_component(SrcName ${SrcRealPath} NAME_WLE)
    get_filename_component(SrcExt ${SrcRealPath} LAST_EXT)
    set(SrcCopy "${PROJECT_BINARY_DIR}/${SrcDir}/${SrcName}${CompileOutSuffix}${SrcExt}")
    add_custom_command(
      OUTPUT ${SrcCopy}
      COMMAND ${CMAKE_COMMAND} -E copy ${Src} ${SrcCopy}
      DEPENDS ${Src}
    )
    list(APPEND PreparedSrc ${SrcCopy})
    set_property(
      SOURCE
      ${SrcCopy}
      APPEND PROPERTY COMPILE_OPTIONS
      ${TARGET_SOURCES_CUSTOM_CUSTOM_FLAGS}
      -I${PROJECT_SOURCE_DIR}/${SrcDir}
    )
  endforeach()

  target_sources(
    ${TgtName}
    PRIVATE
    ${PreparedSrc}
  )
endfunction()

function(generate_enum_serilization Tgt Input)
  set(opts "")
  set(oneval_args INCLUDE_HEADERS GEN_HEADER)
  set(multival_args "")
  cmake_parse_arguments(ENUM_SERIALIZATION_ARGS
    "${opts}"
    "${oneval_args}"
    "${multival_args}"
    ${ARGN}
  )

  get_built_tool_path(enum_parser_bin enum_parser_dependency tools/enum_parser/enum_parser enum_parser)

  get_filename_component(BaseName ${Input} NAME)
  if (ENUM_SERIALIZATION_ARGS_GEN_HEADER)
    set_property(SOURCE ${ENUM_SERIALIZATION_ARGS_GEN_HEADER} PROPERTY GENERATED On)
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${BaseName}_serialized.cpp ${ENUM_SERIALIZATION_ARGS_GEN_HEADER}
      COMMAND
        ${enum_parser_bin}
        ${Input}
        --include-path ${ENUM_SERIALIZATION_ARGS_INCLUDE_HEADERS}
        --output ${CMAKE_CURRENT_BINARY_DIR}/${BaseName}_serialized.cpp
        --header ${ENUM_SERIALIZATION_ARGS_GEN_HEADER}
      DEPENDS ${Input} ${enum_parser_dependency}
    )
  else()
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${BaseName}_serialized.cpp
      COMMAND
        ${enum_parser_bin}
        ${Input}
        --include-path ${ENUM_SERIALIZATION_ARGS_INCLUDE_HEADERS}
        --output ${CMAKE_CURRENT_BINARY_DIR}/${BaseName}_serialized.cpp
      DEPENDS ${Input} ${enum_parser_dependency}
    )
  endif()
  target_sources(${Tgt} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/${BaseName}_serialized.cpp)
endfunction()


if (MSVC AND (${CMAKE_VERSION} VERSION_LESS "3.21.0"))
    message(FATAL_ERROR "Build with MSVC-compatible toolchain requires at least cmake 3.21.0 because of used TARGET_OBJECTS feature")
endif()

function(add_global_library_for TgtName MainName)
  if (MSVC)
    add_library(${TgtName} OBJECT ${ARGN})
    add_dependencies(${TgtName} ${MainName}) # needed because object library can use some extra generated files in MainName
    target_link_libraries(${MainName} INTERFACE ${TgtName} "$<TARGET_OBJECTS:${TgtName}>")
  else()
    add_library(${TgtName} STATIC ${ARGN})
    add_library(${TgtName}.wholearchive INTERFACE)
    add_dependencies(${TgtName}.wholearchive ${TgtName})
    add_dependencies(${TgtName} ${MainName})
    if(APPLE)
      target_link_options(${TgtName}.wholearchive INTERFACE "SHELL:-Wl,-force_load,$<TARGET_FILE:${TgtName}>")
    else()
      target_link_options(${TgtName}.wholearchive INTERFACE "SHELL:-Wl,--whole-archive $<TARGET_FILE:${TgtName}> -Wl,--no-whole-archive")
    endif()
    target_link_libraries(${MainName} INTERFACE ${TgtName}.wholearchive)
  endif()
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
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/vcs_info.json
    COMMAND Python3::Interpreter ${PROJECT_SOURCE_DIR}/build/scripts/generate_vcs_info.py ${CMAKE_CURRENT_BINARY_DIR}/vcs_info.json ${PROJECT_SOURCE_DIR}
    DEPENDS ${PROJECT_SOURCE_DIR}/build/scripts/generate_vcs_info.py
  )

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/__vcs_version__.c
    COMMAND Python3::Interpreter ${PROJECT_SOURCE_DIR}/build/scripts/vcs_info.py ${CMAKE_CURRENT_BINARY_DIR}/vcs_info.json ${CMAKE_CURRENT_BINARY_DIR}/__vcs_version__.c ${PROJECT_SOURCE_DIR}/build/scripts/c_templates/svn_interface.c
    DEPENDS ${PROJECT_SOURCE_DIR}/build/scripts/vcs_info.py ${PROJECT_SOURCE_DIR}/build/scripts/c_templates/svn_interface.c ${CMAKE_CURRENT_BINARY_DIR}/vcs_info.json
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

  get_built_tool_path(rescompiler_bin rescompiler_dependency tools/rescompiler/bin rescompiler)

  add_custom_command(
    OUTPUT ${Output}
    COMMAND ${rescompiler_bin} ${Output} ${ResourcesList}
    DEPENDS ${RESOURCE_ARGS_INPUTS} ${rescompiler_dependency}
  )
endfunction()

function(use_export_script Target ExportFile)
  get_filename_component(OutName ${ExportFile} NAME)
  set(OutPath ${CMAKE_CURRENT_BINARY_DIR}/gen_${OutName})

  if (MSVC)
    target_link_options(${Target} PRIVATE /DEF:${OutPath})
    set(EXPORT_SCRIPT_FLAVOR msvc)
  elseif(APPLE)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/build/scripts/export_script_gen.py ${ExportFile} - --format darwin
      RESULT_VARIABLE _SCRIPT_RES
      OUTPUT_VARIABLE _SCRIPT_FLAGS
      ERROR_VARIABLE _SCRIPT_STDERR
    )
    if (NOT ${_SCRIPT_RES} EQUAL 0)
      message(FATAL_ERROR "Failed to parse export symbols from ${ExportFile}:\n${_SCRIPT_STDERR}")
      return()
    endif()
    separate_arguments(ParsedScriptFlags NATIVE_COMMAND ${_SCRIPT_FLAGS})
    target_link_options(${Target} PRIVATE ${ParsedScriptFlags})
    return()
  else()
    set(EXPORT_SCRIPT_FLAVOR gnu)
    target_link_options(${Target} PRIVATE -Wl,--gc-sections -rdynamic -Wl,--version-script=${OutPath})
  endif()

  add_custom_command(
    OUTPUT ${OutPath}
    COMMAND
      Python3::Interpreter ${PROJECT_SOURCE_DIR}/build/scripts/export_script_gen.py ${ExportFile} ${OutPath} --format ${EXPORT_SCRIPT_FLAVOR}
    DEPENDS ${ExportFile} ${PROJECT_SOURCE_DIR}/build/scripts/export_script_gen.py
  )
  target_sources(${Target} PRIVATE ${OutPath})
  set_property(SOURCE ${OutPath} PROPERTY
    HEADER_FILE_ONLY On
  )
  set_property(TARGET ${Target} APPEND PROPERTY
    LINK_DEPENDS ${OutPath}
  )
endfunction()

function(add_yunittest)
  set(opts "")
  set(oneval_args NAME TEST_TARGET)
  set(multival_args TEST_ARG)
  cmake_parse_arguments(YUNITTEST_ARGS
    "${opts}"
    "${oneval_args}"
    "${multival_args}"
    ${ARGN}
  )

  get_property(SPLIT_FACTOR  TARGET ${YUNITTEST_ARGS_TEST_TARGET} PROPERTY SPLIT_FACTOR)
  get_property(SPLIT_TYPE TARGET ${YUNITTEST_ARGS_TEST_TARGET} PROPERTY SPLIT_TYPE)

  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/run_testpack")
        add_test(NAME ${YUNITTEST_ARGS_NAME} COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/run_testpack" ${YUNITTEST_ARGS_TEST_ARG})
        set_property(TEST ${YUNITTEST_ARGS_NAME} PROPERTY ENVIRONMENT "source_root=${PROJECT_SOURCE_DIR};build_root=${PROJECT_BINARY_DIR};test_split_factor=${SPLIT_FACTOR};test_split_type=${SPLIT_TYPE}")
        return()
  endif()

  if (${SPLIT_FACTOR} EQUAL 1)
  	add_test(NAME ${YUNITTEST_ARGS_NAME} COMMAND ${YUNITTEST_ARGS_TEST_TARGET} ${YUNITTEST_ARGS_TEST_ARG})
  	return()
  endif()

  if ("${SPLIT_TYPE}")
    set(FORK_MODE_ARG --fork-mode ${SPLIT_TYPE})
  endif()
  math(EXPR LastIdx "${SPLIT_FACTOR} - 1")
  foreach(Idx RANGE ${LastIdx})
    add_test(NAME ${YUNITTEST_ARGS_NAME}_${Idx}
      COMMAND Python3::Interpreter ${PROJECT_SOURCE_DIR}/build/scripts/split_unittest.py --split-factor ${SPLIT_FACTOR} ${FORK_MODE_ARG} --shard ${Idx}
       $<TARGET_FILE:${YUNITTEST_ARGS_TEST_TARGET}> ${YUNITTEST_ARGS_TEST_ARG})
  endforeach()
endfunction()

function(set_yunittest_property)
  set(opts "")
  set(oneval_args TEST PROPERTY)
  set(multival_args )
  cmake_parse_arguments(YUNITTEST_ARGS
    "${opts}"
    "${oneval_args}"
    "${multival_args}"
    ${ARGN}
  )
  get_property(SPLIT_FACTOR TARGET ${YUNITTEST_ARGS_TEST} PROPERTY SPLIT_FACTOR)

  if ((${SPLIT_FACTOR} EQUAL 1) OR (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/run_testpack"))
    set_property(TEST ${YUNITTEST_ARGS_TEST} PROPERTY ${YUNITTEST_ARGS_PROPERTY} "${YUNITTEST_ARGS_UNPARSED_ARGUMENTS}")
  	return()
  endif()

  math(EXPR LastIdx "${SPLIT_FACTOR} - 1")
  foreach(Idx RANGE ${LastIdx})
    set_property(TEST ${YUNITTEST_ARGS_TEST}_${Idx} PROPERTY ${YUNITTEST_ARGS_PROPERTY} "${YUNITTEST_ARGS_UNPARSED_ARGUMENTS}")
  endforeach()
endfunction()

option(CUSTOM_ALLOCATORS "Enables use of per executable specified allocators. Can be turned off in order to use code instrumentation tooling relying on system allocator (sanitizers, heaptrack, ...)" On)
function(target_allocator Tgt)
  if (CUSTOM_ALLOCATORS)
    target_link_libraries(${Tgt} PRIVATE ${ARGN})
  else()
    target_link_libraries(${Tgt} PRIVATE system_allocator)
  endif()
endfunction()
