RESOURCES_LIBRARY()



IF (OS_ANDROID)
    # Android SDK for linux: Build-Tools 28.0.0, Build-Tools 28.0.2, Platform 28, Tools 26.1.1
    # DECLARE_EXTERNAL_RESOURCE(ANDROID_SDK sbr:1080226315)
    DECLARE_EXTERNAL_HOST_RESOURCES_BUNDLE(
        ANDROID_SDK
        sbr:1080226315 FOR LINUX
    )
    IF (NOT HOST_OS_LINUX)
        MESSAGE(FATAL_ERROR Unsupported platform for ANDROID_SDK)
    ENDIF()
ELSE()
    MESSAGE(FATAL_ERROR Unsupported platform)
ENDIF()

END()
