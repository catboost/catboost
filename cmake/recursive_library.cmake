function(add_recursive_library Target)
  if (${CMAKE_VERSION} VERSION_LESS "3.21.0")
    message(FATAL_ERROR "add_recursive_library requires at least cmake 3.21.0 (because it uses CXX_LINKER_LAUNCHER)")
  endif()

  if (CMAKE_GENERATOR MATCHES "Visual.Studio.*")
    message(FATAL_ERROR "add_recursive_library is incompatible with Visual Studio generators")
  endif()

  find_package(Python3 REQUIRED)

  # this is not really an executable but we will use it to make CMake collect all dependencies to pass to the custom linking command (because there's no proper way to do it otherwise)
  add_executable(${Target})
  if (NOT (DEFINED CMAKE_POSITION_INDEPENDENT_CODE))
    # default should be the same as for usual static libraries - https://cmake.org/cmake/help/latest/prop_tgt/POSITION_INDEPENDENT_CODE.html
    set_property(TARGET ${Target} PROPERTY POSITION_INDEPENDENT_CODE Off)
  endif()

  set_property(TARGET ${Target} PROPERTY PREFIX ${CMAKE_STATIC_LIBRARY_PREFIX})
  set_property(TARGET ${Target} PROPERTY SUFFIX ${CMAKE_STATIC_LIBRARY_SUFFIX})

  # the result will consist of two files at most (if there are no input files of particular type the resulting output files won't be created):
  #  ${PREFIX}${Target}${SUFFIX} - for objects not requiring global initialization
  #  ${PREFIX}${Target}${GLOBAL_PART_SUFFIX}${SUFFIX} - for objects requiring global initialization
  set(GLOBAL_PART_SUFFIX ".global")

  if (MSVC)
      # if this is not disabled CMake generates additional call to mt.exe after the linking command, manifests are needed only for real executables and dlls
      target_link_options(${Target} PRIVATE "/MANIFEST:NO")
  endif()
  string(CONCAT CXX_LINKER_LAUNCHER_CMD "${Python3_EXECUTABLE}"
    ";${PROJECT_SOURCE_DIR}/build/scripts/create_recursive_library_for_cmake.py"
    ";--project-binary-dir;${PROJECT_BINARY_DIR}"
    ";--cmake-ar;${CMAKE_AR}"
    ";--cmake-ranlib;${CMAKE_RANLIB}"
    ";--cmake-host-system-name;${CMAKE_HOST_SYSTEM_NAME}"
    ";--global-part-suffix;${GLOBAL_PART_SUFFIX}"
  )
  if (CMAKE_CXX_STANDARD_LIBRARIES)
    # because they have to be excluded from input
    string(APPEND CXX_LINKER_LAUNCHER_CMD ";--cmake-cxx-standard-libraries;${CMAKE_CXX_STANDARD_LIBRARIES}")
  endif()
  string(APPEND CXX_LINKER_LAUNCHER_CMD ";--linking-cmdline")  # this must be the last argument

  set_property(TARGET ${Target} PROPERTY CXX_LINKER_LAUNCHER ${CXX_LINKER_LAUNCHER_CMD})
  set_property(TARGET ${Target} PROPERTY LINK_DEPENDS
    "${PROJECT_SOURCE_DIR}/build/scripts/create_recursive_library_for_cmake.py"
    ";${PROJECT_SOURCE_DIR}/build/scripts/link_lib.py"
  )
endfunction()
