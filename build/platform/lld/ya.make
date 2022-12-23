RESOURCES_LIBRARY()

LICENSE(Service-Prebuilt-Tool)



IF (USE_PREVIOUS_LLD_VERSION)
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
ELSE()
    # Use LLD 14
    IF (HOST_OS_LINUX)
        DECLARE_EXTERNAL_RESOURCE(LLD_ROOT sbr:3570346597)
    ELSEIF (HOST_OS_DARWIN)
        IF (HOST_ARCH_ARM64)
            DECLARE_EXTERNAL_RESOURCE(LLD_ROOT sbr:3570338715)
        ELSE()
            DECLARE_EXTERNAL_RESOURCE(LLD_ROOT sbr:3570427908)
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
    # Enable optimized relocations format (e.g. .rela.dyn section) to reduce binary size
    # See:
    # https://android.googlesource.com/platform/ndk/+/master/docs/BuildSystemMaintainers.md#relr-and-relocation-packing
    IF (ANDROID_API >= 30)
        LDFLAGS(-Wl,--pack-dyn-relocs=android+relr)
    ELSEIF (ANDROID_API >= 28)
        LDFLAGS(-Wl,--pack-dyn-relocs=android+relr,--use-android-relr-tags)
    ELSEIF (ANDROID_API >= 23)
        LDFLAGS(-Wl,--pack-dyn-relocs=android)
    ENDIF()
ELSEIF (OS_LINUX)
    LDFLAGS(
        -fuse-ld=${LLD_ROOT_RESOURCE_GLOBAL}/ld.lld

        # dynlinker on auld ubuntu versions can not handle .rodata stored in standalone segment [citation needed]
        -Wl,--no-rosegment
    )
ENDIF()

END()
