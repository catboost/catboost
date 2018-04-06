

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
    resource
    resource/ut
    runtime
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
