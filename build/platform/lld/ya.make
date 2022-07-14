RESOURCES_LIBRARY()

LICENSE(Service-Prebuilt-Tool)



IF (USE_PREVIOUS_LLD_VERSION)
    # Use LLD 11
    IF (HOST_OS_LINUX)
        DECLARE_EXTERNAL_RESOURCE(LLD_ROOT sbr:1843327433)
    ELSEIF (HOST_OS_DARWIN)
        DECLARE_EXTERNAL_RESOURCE(LLD_ROOT sbr:1843327928)
    ENDIF()
ELSE()
    # Use LLD 12
    IF (HOST_OS_LINUX)
        DECLARE_EXTERNAL_RESOURCE(LLD_ROOT sbr:2283360772)
    ELSEIF (HOST_OS_DARWIN)
        IF (HOST_ARCH_ARM64)
            DECLARE_EXTERNAL_RESOURCE(LLD_ROOT sbr:2283439721)
        ELSE()
            DECLARE_EXTERNAL_RESOURCE(LLD_ROOT sbr:2283429958)
        ENDIF()
    ENDIF()
ENDIF()

IF (OS_ANDROID)
    # Use LLD shipped with Android NDK.
    LDFLAGS(
        -fuse-ld=lld
    )
    IF (ANDROID_API < 29)
        # Dynamic linker on Android does not support lld's default rosegment
        # prior to API Level 29 (Android Q)
        # See:
        # https://android.googlesource.com/platform/ndk/+/master/docs/BuildSystemMaintainers.md#additional-required-arguments
        # https://github.com/android/ndk/issues/1196
        LDFLAGS(
            -Wl,--no-rosegment
        )
    ENDIF()
ELSEIF (OS_LINUX)
    LDFLAGS(
        -fuse-ld=${LLD_ROOT_RESOURCE_GLOBAL}/ld.lld

        # dynlinker on auld ubuntu versions can not handle .rodata stored in standalone segment [citation needed]
        -Wl,--no-rosegment
    )
ENDIF()

END()
