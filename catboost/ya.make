

IF (NOT SANITIZER_TYPE STREQUAL "undefined")  # XXX

RECURSE(
    app
    libs
    pytest
    python-package
    R-package
    cuda
    tools
)

ENDIF()
