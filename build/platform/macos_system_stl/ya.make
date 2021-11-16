RESOURCES_LIBRARY()



# Taken from the default toolchain of the Xcode 12.5.1
DECLARE_EXTERNAL_RESOURCE(MACOS_SYSTEM_STL sbr:2561940097)

# xcode toolchain allready contains system headers
IF (USE_STL_SYSTEM AND NOT XCODE)
    CFLAGS(
        GLOBAL -I${MACOS_SYSTEM_STL_RESOURCE_GLOBAL}/include
    )
ENDIF()

END()
