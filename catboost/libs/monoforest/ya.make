LIBRARY()



SRCS(
    additive_model.cpp
    grid.cpp
    helpers.cpp
    model_import.cpp
    monom.cpp
    oblivious_tree.cpp
    polynom.cpp
    split.cpp
)

PEERDIR(
    catboost/libs/helpers
    catboost/libs/logging
    catboost/libs/model
)

END()
