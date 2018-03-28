RESOURCES_LIBRARY()



NO_PLATFORM_RESOURCES()

SET(NEED_PLATFORM_PEERDIRS no)

IF (OS_SDK STREQUAL "local")
    # Implementation is in $S/build/ymake.core.conf
ELSEIF (ARCH_X86_64)
    IF (OS_SDK STREQUAL "ubuntu-10")
        DECLARE_EXTERNAL_RESOURCE(OS_SDK_ROOT sbr:244388930)
    ELSEIF (OS_SDK STREQUAL "ubuntu-12")
        DECLARE_EXTERNAL_RESOURCE(OS_SDK_ROOT sbr:244387436)
    ELSEIF (OS_SDK STREQUAL "ubuntu-14")
        DECLARE_EXTERNAL_RESOURCE(OS_SDK_ROOT sbr:243881007)
    ELSEIF (OS_SDK STREQUAL "ubuntu-16")
        DECLARE_EXTERNAL_RESOURCE(OS_SDK_ROOT sbr:243881345)
    ELSE()
        MESSAGE(FATAL_ERROR "There is no ${OS_SDK} SDK for x86-64")
    ENDIF()
ELSEIF (ARCH_AARCH64)
    IF (OS_SDK STREQUAL "ubuntu-16")
        DECLARE_EXTERNAL_RESOURCE(OS_SDK_ROOT sbr:309054781)
    ELSE()
        MESSAGE(FATAL_ERROR "There is no ${OS_SDK} SDK for aarch64")
    ENDIF()
ELSE()
    MESSAGE(FATAL_ERROR "Unexpected OS_SDK value: ${OS_SDK}")
ENDIF()

# Build uids are not changed when different OS_SDK values are specified.
# Add OS_SDK value to the compile command lines to avoid cache clashes.
# Do not use _OS_SDK preprocessor variable, it will be removed when
# cache is fixed.
CFLAGS(GLOBAL -D_OS_SDK=${OS_SDK})

END()
