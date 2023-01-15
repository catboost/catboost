

RECURSE(
    abseil-cpp
    libffi
)

IF(OS_LINUX OR OS_DARWIN)
    RECURSE(
    
)
ENDIF()

IF(OS_ANDROID)
    RECURSE(
    
)
ENDIF()
