

RECURSE(
    coredump_filter
    coredump_filter/ut
    cores
    filelock
    filelock/ut
    find_root
    func
    func/ut
    fs
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
    runtime_py3
    runtime_py3/main
    runtime_py3/test
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
