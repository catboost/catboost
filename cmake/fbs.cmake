function(target_fbs_source Tgt Key Src)
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
        ${TOOLS_ROOT}/contrib/tools/flatc/bin/flatc
        ${FBS_CPP_FLAGS} ${ARGN}
        -o ${CMAKE_BINARY_DIR}/${fbsRel}.h
        ${Src}
      DEPENDS ${CMAKE_SOURCE_DIR}/build/scripts/cpp_flatc_wrapper.py ${Src} ${TOOLS_ROOT}/contrib/tools/flatc/bin/flatc
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    target_sources(${Tgt} ${Key}
      ${CMAKE_BINARY_DIR}/${fbsRel}.cpp
      ${CMAKE_BINARY_DIR}/${fbsRel}.h
      ${OutputDir}/${OutputBase}.iter.fbs.h
    )
endfunction()
