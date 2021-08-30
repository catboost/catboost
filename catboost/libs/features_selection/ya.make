LIBRARY()



SRCS(
    recursive_features_elimination.cpp
    select_features.cpp
    selection_results.cpp
)

PEERDIR(
    catboost/libs/data
    catboost/libs/eval_result
    catboost/libs/fstr
    catboost/libs/model
    catboost/libs/helpers
    catboost/libs/train_lib
    catboost/private/libs/algo
    catboost/private/libs/distributed
    catboost/private/libs/labels
    catboost/private/libs/options
    catboost/private/libs/target
    library/cpp/json/writer
)

END()
