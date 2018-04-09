

RECURSE(
    filelock
    filelock/ut
    find_root
    func
    func/ut
    pytest
    pytest/allure
    pytest/empty
    pytest/plugins
    runtime
    runtime/main
    strings
    strings/ut
    testing
    windows
    windows/ut
)

IF (OS_LINUX)
    RECURSE(
    
)
ENDIF()

IF (NOT MUSL)
    RECURSE(
    
)
ENDIF()
