RESOURCES_LIBRARY()



IF (OS_ANDROID)
    # Android SDK for darwin: Build-Tools 30.0.3, Platform 30 with upgraded emulator and armv8-arm64 system image
    IF(ARCH_ARM64 AND HOST_OS_DARWIN)
        DECLARE_EXTERNAL_HOST_RESOURCES_BUNDLE(
            ANDROID_SDK
            sbr:2966636073 FOR DARWIN
        )
    ELSE()
        # Android SDK for linux and darwin: Build-Tools 30.0.3, Platform 30
        DECLARE_EXTERNAL_HOST_RESOURCES_BUNDLE(
            ANDROID_SDK
            sbr:2564045529 FOR LINUX
            sbr:2564523615 FOR DARWIN
        )
    ENDIF()
    IF (NOT HOST_OS_LINUX AND NOT HOST_OS_DARWIN)
        MESSAGE(FATAL_ERROR Unsupported platform for ANDROID_SDK)
    ENDIF()
    DECLARE_EXTERNAL_RESOURCE(ANDROID_AVD sbr:2965845602)
ELSE()
    MESSAGE(FATAL_ERROR Unsupported platform)
ENDIF()

END()
