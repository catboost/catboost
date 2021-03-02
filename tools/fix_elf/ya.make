

IF (USE_PREBUILT_TOOLS)
    INCLUDE(${ARCADIA_ROOT}/build/prebuilt/tools/fix_elf/ya.make.prebuilt)
ENDIF()

IF (NOT PREBUILT)
    INCLUDE(${ARCADIA_ROOT}/tools/fix_elf/bin/ya.make)
ENDIF()

RECURSE(
    bin
)
