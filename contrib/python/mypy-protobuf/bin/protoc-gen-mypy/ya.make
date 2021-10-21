

IF (USE_PREBUILT_TOOLS)
    INCLUDE(${ARCADIA_ROOT}/build/prebuilt/contrib/python/mypy-protobuf/bin/protoc-gen-mypy/ya.make.prebuilt)
ENDIF()

IF (NOT PREBUILT)
    INCLUDE(${ARCADIA_ROOT}/contrib/python/mypy-protobuf/bin/protoc-gen-mypy/bin/ya.make)
ENDIF()

RECURSE(
    bin
)
