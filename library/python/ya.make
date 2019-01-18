

RECURSE(
    cores
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
    reservoir_sampling
    resource
    runtime
    runtime/main
    runtime/test
    strings
    strings/ut
    symbols
    testing
    windows
    windows/ut
)

IF (NOT MUSL)
    RECURSE(
    
)
ENDIF()
