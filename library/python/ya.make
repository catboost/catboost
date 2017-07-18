

RECURSE(
    find_root
    pytest
    pytest/allure
    runtime
    testing
)

IF (OS_LINUX)
    RECURSE(
    
)
ENDIF()

IF (NOT MUSL)
    RECURSE(
    
)
ENDIF()
