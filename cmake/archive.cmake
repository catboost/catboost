find_package(Python3 REQUIRED)

function(target_rodata_sources TgtName Scope)
  foreach(rodata ${ARGN})
    get_filename_component(CppVar ${rodata} NAME_WLE)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${CppVar}.cpp
        COMMAND Python3::Interpreter ${PROJECT_SOURCE_DIR}/build/scripts/rodata2cpp.py ${CppVar} ${rodata} ${CMAKE_CURRENT_BINARY_DIR}/${CppVar}.cpp
        DEPENDS ${PROJECT_SOURCE_DIR}/build/scripts/rodata2cpp.py ${rodata}
    )
    target_sources(${TgtName} ${Scope} ${CMAKE_CURRENT_BINARY_DIR}/${CppVar}.cpp)
  endforeach()
endfunction()