

IF (NOT SANITIZER_TYPE STREQUAL "undefined")  # XXX

RECURSE(
    R-package
    app
    idl
    jvm-packages
    libs
    private
    pytest
    python-package
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
