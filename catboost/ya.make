

IF (SANITIZER_TYPE != "undefined")  # XXX

IF (EXPORT_CMAKE AND OS_ANDROID)
    RECURSE(
    libs/model_interface
)
ELSE()
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
ENDIF()

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
