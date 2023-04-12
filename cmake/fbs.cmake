include(common)

function(target_fbs_source Tgt Key Src)
    get_built_tool_path(flatc_bin flatc_dependency contrib/tools/flatc/bin  flatc)

    file(RELATIVE_PATH fbsRel ${CMAKE_SOURCE_DIR} ${Src})
    get_filename_component(OutputBase ${fbsRel} NAME_WLE)
    get_filename_component(OutputDir ${CMAKE_BINARY_DIR}/${fbsRel} DIRECTORY)
    add_custom_command(
      OUTPUT
        ${CMAKE_BINARY_DIR}/${fbsRel}.h
        ${CMAKE_BINARY_DIR}/${fbsRel}.cpp
        ${OutputDir}/${OutputBase}.iter.fbs.h
        ${OutputDir}/${OutputBase}.bfbs
      COMMAND Python3::Interpreter
        ${CMAKE_SOURCE_DIR}/build/scripts/cpp_flatc_wrapper.py
        ${flatc_bin}
        ${FBS_CPP_FLAGS} ${ARGN}
        -o ${CMAKE_BINARY_DIR}/${fbsRel}.h
        ${Src}
      DEPENDS ${CMAKE_SOURCE_DIR}/build/scripts/cpp_flatc_wrapper.py ${Src} ${flatc_dependency}
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    target_sources(${Tgt} ${Key}
      ${CMAKE_BINARY_DIR}/${fbsRel}.cpp
      ${CMAKE_BINARY_DIR}/${fbsRel}.h
      ${OutputDir}/${OutputBase}.iter.fbs.h
    )
endfunction()
