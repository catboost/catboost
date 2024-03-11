include(common)

function(target_fbs_source Tgt Key Src)
    get_built_tool_path(flatc_bin flatc_dependency contrib/tools/flatc/bin  flatc)

    file(RELATIVE_PATH fbsRel ${PROJECT_SOURCE_DIR} ${Src})
    get_filename_component(OutputBase ${fbsRel} NAME_WLE)
    get_filename_component(OutputDir ${PROJECT_BINARY_DIR}/${fbsRel} DIRECTORY)
    add_custom_command(
      OUTPUT
        ${PROJECT_BINARY_DIR}/${fbsRel}.h
        ${PROJECT_BINARY_DIR}/${fbsRel}.cpp
        ${OutputDir}/${OutputBase}.iter.fbs.h
        ${OutputDir}/${OutputBase}.bfbs
      COMMAND Python3::Interpreter
        ${PROJECT_SOURCE_DIR}/build/scripts/cpp_flatc_wrapper.py
        ${flatc_bin}
        ${FBS_CPP_FLAGS} ${ARGN}
        -o ${PROJECT_BINARY_DIR}/${fbsRel}.h
        ${Src}
      DEPENDS ${PROJECT_SOURCE_DIR}/build/scripts/cpp_flatc_wrapper.py ${Src} ${flatc_dependency}
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    )
    target_sources(${Tgt} ${Key}
      ${PROJECT_BINARY_DIR}/${fbsRel}.cpp
      ${PROJECT_BINARY_DIR}/${fbsRel}.h
      ${OutputDir}/${OutputBase}.iter.fbs.h
    )
endfunction()
