

IF (NOT SANITIZER_TYPE STREQUAL "undefined")  # XXX

RECURSE(
    app
    cuda
    idl
    libs
    pytest
    python-package
    R-package
    tools
)

IF (NOT CATBOOST_OPENSOURCE)
RECURSE(
    
)
ENDIF()

ENDIF()
