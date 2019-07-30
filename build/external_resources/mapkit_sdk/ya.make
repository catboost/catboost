RESOURCES_LIBRARY()



IF (GCC OR CLANG)
    # headers must be fixed later
    CFLAGS(
        GLOBAL "-Wno-error=unused-parameter"
        GLOBAL "-Wno-error=sign-compare"
    )
ENDIF()

IF (OS_LINUX)
    # Qt 5.6.1 + protobuf 3.6.1 + GL headers + GLES2
    DECLARE_EXTERNAL_RESOURCE(MAPKIT_SDK sbr:897945809)
    CFLAGS(
        GLOBAL "-I$MAPKIT_SDK_RESOURCE_GLOBAL/mapkit_sdk/include"
        GLOBAL "-I$MAPKIT_SDK_RESOURCE_GLOBAL/mapkit_sdk/local/linux.x86-64/include"
    )
    LDFLAGS_FIXED(
        "-L$MAPKIT_SDK_RESOURCE_GLOBAL/mapkit_sdk/local/linux.x86-64/lib"
        "-L$MAPKIT_SDK_RESOURCE_GLOBAL/mapkit_sdk/lib/x86_64-linux-gnu"
    )
ELSEIF (OS_ANDROID)
    # protobuf 3.6.1
    DECLARE_EXTERNAL_RESOURCE(MAPKIT_SDK sbr:881642915)
    DECLARE_EXTERNAL_RESOURCE(MAPS_NDK_PATCH sbr:1045044111)
    CFLAGS(
        GLOBAL "-I$MAPKIT_SDK_RESOURCE_GLOBAL/mapkit_sdk/include"
    )
    IF (ARCH_ARM7)
        LDFLAGS_FIXED(
            "-L$MAPS_NDK_PATCH/android.armeabi-v7a/lib"
            "-L$MAPKIT_SDK_RESOURCE_GLOBAL/mapkit_sdk/local/android.armeabi-v7a/lib"
        )
        CFLAGS(
            GLOBAL "-I$MAPS_NDK_PATCH/android.armeabi-v7a/include"
        )
    ELSEIF (ARCH_ARM64)
        LDFLAGS_FIXED(
            "-L$MAPS_NDK_PATCH/android.arm64-v8a/lib"
            "-L$MAPKIT_SDK_RESOURCE_GLOBAL/mapkit_sdk/local/android.arm64-v8a/lib"
        )
        CFLAGS(
            GLOBAL "-I$MAPS_NDK_PATCH/android.arm64-v8a/include"
        )
    ELSEIF(ARCH_I386)
        LDFLAGS_FIXED(
            "-L$MAPS_NDK_PATCH/android.x86/lib"
            "-L$MAPKIT_SDK_RESOURCE_GLOBAL/mapkit_sdk/local/android.x86/lib"
        )
        CFLAGS(
            GLOBAL "-I$MAPS_NDK_PATCH/android.x86/include"
        )
    ELSEIF (ARCH_X86_64)
        LDFLAGS_FIXED(
            "-L$MAPS_NDK_PATCH/android.x86_64/lib"
            "-L$MAPKIT_SDK_RESOURCE_GLOBAL/mapkit_sdk/local/android.x86_64/lib"
        )
        CFLAGS(
            GLOBAL "-I$MAPS_NDK_PATCH/android.x86_64/include"
        )
    ELSE()
        MESSAGE(FATAL_ERROR Unsupported platform)
    ENDIF()
ELSEIF (OS_DARWIN)
    # Qt + protobuf 2.6.1
    DECLARE_EXTERNAL_RESOURCE(MAPKIT_SDK sbr:666723854)
    CFLAGS(
        GLOBAL "-I$MAPKIT_SDK_RESOURCE_GLOBAL/mapkit_sdk/include"
        GLOBAL "-I$MAPKIT_SDK_RESOURCE_GLOBAL/mapkit_sdk/local/darwin.x86-64/include"
    )
    LDFLAGS_FIXED(
        "-L$MAPKIT_SDK_RESOURCE_GLOBAL/mapkit_sdk/local/darwin.x86-64/lib"
    )
ELSEIF (OS_IOS)
    # protobuf 2.6.1
    IF (HOST_OS_LINUX)
        DECLARE_EXTERNAL_RESOURCE(MAPKIT_SDK sbr:666724415)
    ELSEIF (HOST_OS_DARWIN)
        DECLARE_EXTERNAL_RESOURCE(MAPKIT_SDK sbr:731932280)
    ELSE()
        MESSAGE(FATAL_ERROR Unsupported platform)
    ENDIF()
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
