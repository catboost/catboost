LIBRARY()



SRCS(
    grid_creator.cpp
    utils.cpp
)

PEERDIR(
    library/cpp/grid_creator
    library/cpp/threading/local_executor
    catboost/libs/helpers
    catboost/private/libs/options
)

END()
