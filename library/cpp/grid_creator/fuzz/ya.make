

FUZZ()

FUZZ_OPTS(
    -max_len=1024
    -use_counters=1
    -use_value_profile=1
)

SRCS(
    main.cpp
)

PEERDIR(
    library/cpp/grid_creator
)

END()
