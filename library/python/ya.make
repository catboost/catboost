

RECURSE(
    filelock
    filelock/ut
    find_root
    func
    func/ut
    pymain
    pytest
    pytest/allure
    pytest/empty
    pytest/plugins
    resource
    resource/ut
    runtime
    runtime/main
    runtime/test
    runtime_py3
    runtime_py3/main
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

NEED_CHECK()
