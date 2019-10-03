LIBRARY()



SRCS(
    grid_creator.cpp
    utils.cpp
)

PEERDIR(
    library/grid_creator
    library/threading/local_executor
    catboost/libs/helpers
    catboost/private/libs/options
)

END()
