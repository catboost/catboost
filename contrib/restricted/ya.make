

RECURSE(
    abseil-cpp
    libffi
    murmurhash
)

IF(OS_LINUX OR OS_DARWIN)
    RECURSE(
    
)
ENDIF()

IF(OS_ANDROID)
    RECURSE(
    
)
ENDIF()
