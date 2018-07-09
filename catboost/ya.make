

IF (NOT SANITIZER_TYPE STREQUAL "undefined")  # XXX

RECURSE(
    app
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

IF (HAVE_CUDA)
RECURSE(
    cuda
)
ENDIF()

ENDIF()
