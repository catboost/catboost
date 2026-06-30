add_custom_target(all-shared-libs)

function(add_shared_library Tgt)
  add_library(${Tgt} SHARED ${ARGN})
  add_dependencies(all-shared-libs ${Tgt})
  if (NOT CMAKE_POSITION_INDEPENDENT_CODE)
    set_property(TARGET ${Tgt} PROPERTY EXCLUDE_FROM_ALL On)
  endif()
endfunction()
