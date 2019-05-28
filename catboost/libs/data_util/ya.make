LIBRARY()




SRCS(
    GLOBAL line_data_reader.cpp
    GLOBAL exists_checker.cpp
    path_with_scheme.cpp
)

PEERDIR(
    catboost/libs/index_range
    library/binsaver
    library/object_factory
)

END()
