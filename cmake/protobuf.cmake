include(common)

function(target_proto_plugin Tgt Name PluginTarget)
  set_property(TARGET ${Tgt} APPEND PROPERTY
    PROTOC_OPTS --${Name}_out=${PROJECT_BINARY_DIR}/$<TARGET_PROPERTY:${Tgt},PROTO_NAMESPACE> --plugin=protoc-gen-${Name}=$<TARGET_FILE:${PluginTarget}>
  )
  set_property(TARGET ${Tgt} APPEND PROPERTY
    PROTOC_DEPS ${PluginTarget}
  )
endfunction()

function(target_proto_addincls Tgt)
  set_property(TARGET ${Tgt} APPEND PROPERTY
    PROTO_ADDINCL ${ARGN}
  )
endfunction()

function(target_proto_outs Tgt)
  set_property(TARGET ${Tgt} APPEND PROPERTY
    PROTO_OUTS ${ARGN}
  )
endfunction()

function(target_messages Tgt Scope UseStyleguide UseEvent2Cpp)
  if (vanilla_protobuf STREQUAL "yes")
    set(protoc_bin ${PROJECT_BINARY_DIR}/bin/protoc${CMAKE_EXECUTABLE_SUFFIX})
    set(protoc_dependency "")
    set(UseStyleguide "no")  # cpp_styleguide can't compile with vanilla protobuf
  else()
    get_built_tool_path(protoc_bin protoc_dependency "contrib/tools/protoc/bin" "protoc")
  endif()

  if (UseStyleguide STREQUAL "yes")
    get_built_tool_path(cpp_styleguide_bin cpp_styleguide_dependency "contrib/tools/protoc/plugins/cpp_styleguide" "cpp_styleguide")
    set(protoc_styleguide_plugin --plugin=protoc-gen-cpp_styleguide=${cpp_styleguide_bin})
  else()
    set(protoc_styleguide_plugin "")
    set(cpp_styleguide_dependency "")
  endif()

  if (UseEvent2Cpp STREQUAL "yes")
    get_built_tool_path(event2cpp_bin event2cpp_dependency "tools/event2cpp/bin" "event2cpp")
    set(protoc_event2cpp_plugin --plugin=protoc-gen-event2cpp=${event2cpp_bin})
    set(ext_h ".ev.pb.h")
    set(ext_c ".ev.pb.cc")
  else()
    set(protoc_event2cpp_plugin "")
    set(event2cpp_dependency "")
    set(ext_h ".pb.h")
    set(ext_c ".pb.cc")
  endif()

  get_property(ProtocExtraOutsSuf TARGET ${Tgt} PROPERTY PROTOC_EXTRA_OUTS)
  foreach(proto ${ARGN})
    if(proto MATCHES ${PROJECT_BINARY_DIR})
      file(RELATIVE_PATH protoRel ${PROJECT_BINARY_DIR} ${proto})
    elseif (proto MATCHES ${PROJECT_SOURCE_DIR})
      file(RELATIVE_PATH protoRel ${PROJECT_SOURCE_DIR} ${proto})
    else()
      set(protoRel ${proto})
    endif()
    get_filename_component(OutputBase ${protoRel} NAME_WLE)
    get_filename_component(OutputDir ${PROJECT_BINARY_DIR}/${protoRel} DIRECTORY)
    list(TRANSFORM ProtocExtraOutsSuf PREPEND ${OutputDir}/${OutputBase} OUTPUT_VARIABLE ProtocExtraOuts)
    add_custom_command(
        OUTPUT
          ${OutputDir}/${OutputBase}${ext_c}
          ${OutputDir}/${OutputBase}${ext_h}
          ${ProtocExtraOuts}
        COMMAND ${protoc_bin}
          ${COMMON_PROTOC_FLAGS}
          "-I$<JOIN:$<TARGET_GENEX_EVAL:${Tgt},$<TARGET_PROPERTY:${Tgt},PROTO_ADDINCL>>,;-I>"
          "$<JOIN:$<TARGET_GENEX_EVAL:${Tgt},$<TARGET_PROPERTY:${Tgt},PROTO_OUTS>>,;>"
          ${protoc_styleguide_plugin}
          ${protoc_event2cpp_plugin}
          "$<JOIN:$<TARGET_GENEX_EVAL:${Tgt},$<TARGET_PROPERTY:${Tgt},PROTOC_OPTS>>,;>"
          ${protoRel}
        COMMAND Python3::Interpreter ${PROJECT_SOURCE_DIR}/build/scripts/re_replace.py --from-re "\"((?:struct|class)\\s+\\S+\\s+)final\\s*:\"" --to-re "\"\\1:\"" ${OutputDir}/${OutputBase}${ext_c} ${OutputDir}/${OutputBase}${ext_h}
        DEPENDS
          ${proto}
          $<TARGET_PROPERTY:${Tgt},PROTOC_DEPS>
          ${protoc_dependency}
          ${cpp_styleguide_dependency}
          ${event2cpp_dependency}
          ${PROJECT_SOURCE_DIR}/build/scripts/re_replace.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMAND_EXPAND_LISTS
    )
    target_sources(${Tgt} ${Scope}
      ${OutputDir}/${OutputBase}${ext_c} ${OutputDir}/${OutputBase}${ext_h}
      ${ProtocExtraOuts}
    )
  endforeach()
endfunction()

function(target_proto_messages Tgt Scope)
  target_messages(${Tgt} ${Scope} "yes" "no" ${ARGN})
endfunction()

function(target_ev_messages Tgt Scope)
  target_messages(${Tgt} ${Scope} "yes" "yes" ${ARGN})
endfunction()
