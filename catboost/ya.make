

IF (NOT SANITIZER_TYPE STREQUAL "undefined")  # XXX

RECURSE(
    app
    cuda
    libs
    pytest
    python-package
    R-package
    tools
)

ENDIF()
