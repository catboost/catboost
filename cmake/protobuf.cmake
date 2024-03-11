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

function(target_proto_messages Tgt Scope)
  get_built_tool_path(protoc_bin protoc_dependency contrib/tools/protoc/bin protoc)
  get_built_tool_path(cpp_styleguide_bin cpp_styleguide_dependency contrib/tools/protoc/plugins/cpp_styleguide cpp_styleguide)

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
          ${OutputDir}/${OutputBase}.pb.cc
          ${OutputDir}/${OutputBase}.pb.h
          ${ProtocExtraOuts}
        COMMAND ${protoc_bin}
          ${COMMON_PROTOC_FLAGS}
          "-I$<JOIN:$<TARGET_GENEX_EVAL:${Tgt},$<TARGET_PROPERTY:${Tgt},PROTO_ADDINCL>>,;-I>"
          "$<JOIN:$<TARGET_GENEX_EVAL:${Tgt},$<TARGET_PROPERTY:${Tgt},PROTO_OUTS>>,;>"
          --plugin=protoc-gen-cpp_styleguide=${cpp_styleguide_bin}
          "$<JOIN:$<TARGET_GENEX_EVAL:${Tgt},$<TARGET_PROPERTY:${Tgt},PROTOC_OPTS>>,;>"
          ${protoRel}
        DEPENDS
          ${proto}
          $<TARGET_PROPERTY:${Tgt},PROTOC_DEPS>
          ${protoc_dependency}
          ${cpp_styleguide_dependency}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMAND_EXPAND_LISTS
    )
    target_sources(${Tgt} ${Scope}
      ${OutputDir}/${OutputBase}.pb.cc ${OutputDir}/${OutputBase}.pb.h
      ${ProtocExtraOuts}
    )
  endforeach()
endfunction()

function(target_ev_messages Tgt Scope)
  get_built_tool_path(protoc_bin protoc_dependency contrib/tools/protoc/bin protoc)
  get_built_tool_path(cpp_styleguide_bin cpp_styleguide_dependency contrib/tools/protoc/plugins/cpp_styleguide cpp_styleguide)
  get_built_tool_path(event2cpp_bin event2cpp_dependency tools/event2cpp/bin event2cpp)

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
          ${OutputDir}/${OutputBase}.ev.pb.cc
          ${OutputDir}/${OutputBase}.ev.pb.h
          ${ProtocExtraOuts}
        COMMAND ${protoc_bin}
          ${COMMON_PROTOC_FLAGS}
          "-I$<JOIN:$<TARGET_GENEX_EVAL:${Tgt},$<TARGET_PROPERTY:${Tgt},PROTO_ADDINCL>>,;-I>"
          "$<JOIN:$<TARGET_GENEX_EVAL:${Tgt},$<TARGET_PROPERTY:${Tgt},PROTO_OUTS>>,;>"
          --plugin=protoc-gen-cpp_styleguide=${cpp_styleguide_bin}
          --plugin=protoc-gen-event2cpp=${event2cpp_bin}
          "$<JOIN:$<TARGET_GENEX_EVAL:${Tgt},$<TARGET_PROPERTY:${Tgt},PROTOC_OPTS>>,;>"
          ${protoRel}
        DEPENDS
          ${proto}
          $<TARGET_PROPERTY:${Tgt},PROTOC_DEPS>
          ${protoc_dependency}
          ${cpp_styleguide_dependency}
          ${event2cpp_dependency}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMAND_EXPAND_LISTS
    )
    target_sources(${Tgt} ${Scope}
      ${OutputDir}/${OutputBase}.ev.pb.cc ${OutputDir}/${OutputBase}.ev.pb.h
      ${ProtocExtraOuts}
    )
  endforeach()
endfunction()
