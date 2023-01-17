

IF (CUDA_VERSION VERSION_GT 11) 
    RECURSE(
    
)
ENDIF()

RECURSE(
    cub
    thrust
)
