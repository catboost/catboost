

RECURSE(
    medium
)

IF (HAVE_CUDA)
    RECURSE(
    medium/gpu
)
ENDIF()
