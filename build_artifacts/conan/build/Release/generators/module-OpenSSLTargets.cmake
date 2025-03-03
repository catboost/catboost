# Load the debug and release variables
file(GLOB DATA_FILES "${CMAKE_CURRENT_LIST_DIR}/module-OpenSSL-*-data.cmake")

foreach(f ${DATA_FILES})
    include(${f})
endforeach()

# Create the targets for all the components
foreach(_COMPONENT ${openssl_COMPONENT_NAMES} )
    if(NOT TARGET ${_COMPONENT})
        add_library(${_COMPONENT} INTERFACE IMPORTED)
        message(${OpenSSL_MESSAGE_MODE} "Conan: Component target declared '${_COMPONENT}'")
    endif()
endforeach()

if(NOT TARGET openssl::openssl)
    add_library(openssl::openssl INTERFACE IMPORTED)
    message(${OpenSSL_MESSAGE_MODE} "Conan: Target declared 'openssl::openssl'")
endif()
# Load the debug and release library finders
file(GLOB CONFIG_FILES "${CMAKE_CURRENT_LIST_DIR}/module-OpenSSL-Target-*.cmake")

foreach(f ${CONFIG_FILES})
    include(${f})
endforeach()