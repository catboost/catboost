########## MACROS ###########################################################################
#############################################################################################

# Requires CMake > 3.15
if(${CMAKE_VERSION} VERSION_LESS "3.15")
    message(FATAL_ERROR "The 'CMakeDeps' generator only works with CMake >= 3.15")
endif()

if(OpenSSL_FIND_QUIETLY)
    set(OpenSSL_MESSAGE_MODE VERBOSE)
else()
    set(OpenSSL_MESSAGE_MODE STATUS)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/cmakedeps_macros.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/OpenSSLTargets.cmake)
include(CMakeFindDependencyMacro)

check_build_type_defined()

foreach(_DEPENDENCY ${openssl_FIND_DEPENDENCY_NAMES} )
    # Check that we have not already called a find_package with the transitive dependency
    if(NOT ${_DEPENDENCY}_FOUND)
        find_dependency(${_DEPENDENCY} REQUIRED ${${_DEPENDENCY}_FIND_MODE})
    endif()
endforeach()

set(OpenSSL_VERSION_STRING "1.1.1t")
set(OpenSSL_INCLUDE_DIRS ${openssl_INCLUDE_DIRS_RELEASE} )
set(OpenSSL_INCLUDE_DIR ${openssl_INCLUDE_DIRS_RELEASE} )
set(OpenSSL_LIBRARIES ${openssl_LIBRARIES_RELEASE} )
set(OpenSSL_DEFINITIONS ${openssl_DEFINITIONS_RELEASE} )


# Only the last installed configuration BUILD_MODULES are included to avoid the collision
foreach(_BUILD_MODULE ${openssl_BUILD_MODULES_PATHS_RELEASE} )
    message(${OpenSSL_MESSAGE_MODE} "Conan: Including build module from '${_BUILD_MODULE}'")
    include(${_BUILD_MODULE})
endforeach()


