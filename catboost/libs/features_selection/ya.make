LIBRARY()



SRCS(
    recursive_features_elimination.cpp
    select_features.cpp
    selection_results.cpp
)

PEERDIR(
    catboost/libs/data
    catboost/libs/helpers
    catboost/libs/train_lib
)

END()
