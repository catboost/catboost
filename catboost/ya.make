

IF (SANITIZER_TYPE != "undefined")  # XXX

RECURSE(
    R-package
    app
    idl
    jvm-packages
    libs
    private
    pytest
    python-package
    spark
    tools
    docs
)

IF (NOT OPENSOURCE)
RECURSE(
    
)
ENDIF()

IF (HAVE_CUDA)
RECURSE(
    cuda
)
ENDIF()

ENDIF()
