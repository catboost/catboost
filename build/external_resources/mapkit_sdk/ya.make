RESOURCES_LIBRARY()



IF (OS_LINUX)
    # Qt + protobuf 2.6.1
    DECLARE_EXTERNAL_RESOURCE(MAPKIT_SDK sbr:524712825)
    CFLAGS(
        GLOBAL "-I$MAPKIT_SDK_RESOURCE_GLOBAL/mapkit_sdk/include"
        GLOBAL "-I$MAPKIT_SDK_RESOURCE_GLOBAL/mapkit_sdk/local/linux.x86-64/include"
    )
    LDFLAGS_FIXED(
        "-L$MAPKIT_SDK_RESOURCE_GLOBAL/mapkit_sdk/local/linux.x86-64/lib"
    )
ELSEIF (OS_ANDROID)
    # protobuf 2.6.1
    DECLARE_EXTERNAL_RESOURCE(MAPKIT_SDK sbr:549833385)
    CFLAGS(
        GLOBAL "-I$MAPKIT_SDK_RESOURCE_GLOBAL/mapkit_sdk/include"
    )
    IF (ARCH_ARM7)
        LDFLAGS_FIXED(
            "-L$MAPKIT_SDK_RESOURCE_GLOBAL/mapkit_sdk/local/android.armeabi-v7a/lib"
        )
    ELSEIF(ARCH_I386)
        LDFLAGS_FIXED(
            "-L$MAPKIT_SDK_RESOURCE_GLOBAL/mapkit_sdk/local/android.x86/lib"
        )
    ELSE()
        MESSAGE(FATAL_ERROR Unsupported platform)
    ENDIF()
ELSE()
    MESSAGE(FATAL_ERROR Unsupported platform)
ENDIF()

END()
