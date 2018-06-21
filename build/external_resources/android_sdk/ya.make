RESOURCES_LIBRARY()



IF (OS_ANDROID)
    # Andrid SDK + build tools 23.0.3 + build tools 27.0.0 + api level 23 + api level 27
    DECLARE_EXTERNAL_RESOURCE(ANDROID_SDK sbr:580754955)
ELSE()
    MESSAGE(FATAL_ERROR Unsupported platform)
ENDIF()

END()

