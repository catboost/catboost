RESOURCES_LIBRARY()



IF (OS_LINUX)
    # Qt + protobuf 2.6.1 + GL headers + GLES2
    DECLARE_EXTERNAL_RESOURCE(MAPKIT_SDK sbr:648642209)
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
        CFLAGS(
            GLOBAL "-I$MAPKIT_SDK_RESOURCE_GLOBAL/mapkit_sdk/arch-x86/usr/include"
        )
        LDFLAGS_FIXED(
            "-L$MAPKIT_SDK_RESOURCE_GLOBAL/mapkit_sdk/local/android.x86/lib"
            "-L$MAPKIT_SDK_RESOURCE_GLOBAL/mapkit_sdk/arch-x86/usr/lib"
        )
    ELSE()
        MESSAGE(FATAL_ERROR Unsupported platform)
    ENDIF()
ELSEIF (OS_DARWIN)
    # Qt + protobuf 2.6.1
    DECLARE_EXTERNAL_RESOURCE(MAPKIT_SDK sbr:641729919)
    CFLAGS(
        GLOBAL "-I$MAPKIT_SDK_RESOURCE_GLOBAL/mapkit_sdk/include"
        GLOBAL "-I$MAPKIT_SDK_RESOURCE_GLOBAL/mapkit_sdk/local/darwin.x86-64/include"
    )
    LDFLAGS_FIXED(
        "-L$MAPKIT_SDK_RESOURCE_GLOBAL/mapkit_sdk/local/darwin.x86-64/lib"
    )
ELSEIF (OS_IOS)
    # protobuf 2.6.1
    DECLARE_EXTERNAL_RESOURCE(MAPKIT_SDK sbr:641730454)
    CFLAGS(
        GLOBAL "-I$MAPKIT_SDK_RESOURCE_GLOBAL/mapkit_sdk/include"
    )
    IF (ARCH_ARM64)
        LDFLAGS_FIXED(
            "-L$MAPKIT_SDK_RESOURCE_GLOBAL/mapkit_sdk/local/ios.arm64/lib"
        )
    ELSEIF (ARCH_ARM7)
        LDFLAGS_FIXED(
            "-L$MAPKIT_SDK_RESOURCE_GLOBAL/mapkit_sdk/local/ios.armv7/lib"
        )
    ELSEIF (ARCH_I386)
        LDFLAGS_FIXED(
            "-L$MAPKIT_SDK_RESOURCE_GLOBAL/mapkit_sdk/local/ios.i386/lib"
        )
    ELSEIF (ARCH_X86_64)
        LDFLAGS_FIXED(
            "-L$MAPKIT_SDK_RESOURCE_GLOBAL/mapkit_sdk/local/ios.x86-64/lib"
        )
    ELSE()
        MESSAGE(FATAL_ERROR Unsupported platform)
    ENDIF()
ELSE()
    MESSAGE(FATAL_ERROR Unsupported platform)
ENDIF()

END()
