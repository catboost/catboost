RESOURCES_LIBRARY()



SET(NEED_PLATFORM_PEERDIRS no)

IF (ARCH_ARM7 OR ARCH_ARM64)
    # iOS 13.0 SDK / Xcode 11.0 (11A420a)
    DECLARE_EXTERNAL_RESOURCE(IOS_SDK_ROOT sbr:1137289555)
ELSEIF (ARCH_I386 OR ARCH_X86_64)
    # iOS Simulator 13.0 SDK / Xcode 11.0 (11A420a)
    DECLARE_EXTERNAL_RESOURCE(IOS_SDK_ROOT sbr:1137292304)
ELSE()
    MESSAGE(FATAL_ERROR "There is no iOS SDK for the selected target platform")
ENDIF()

# Apple C++ standard library headers are not in OS SDK but in Xcode toolchain
IF (USE_STL_SYSTEM)
    SET(__XCODE_RESOURCE_NAME CPP_XCODE_TOOLCHAIN_ROOT)
    SET(__XCODE_TOOLCHAIN_VERSION 10.2.1)
    INCLUDE(${ARCADIA_ROOT}/build/platform/xcode/ya.make.inc)
    IF (OS_IOS)
        CFLAGS(
            GLOBAL -cxx-isystem GLOBAL $CPP_XCODE_TOOLCHAIN_ROOT_RESOURCE_GLOBAL/usr/include/c++/v1
            GLOBAL -cxx-isystem GLOBAL $CPP_XCODE_TOOLCHAIN_ROOT_RESOURCE_GLOBAL/usr/include
        )
    ENDIF()
ENDIF()

END()
