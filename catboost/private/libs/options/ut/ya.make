UNITTEST(json_options_ut)



SRCS(
    json_helper_ut.cpp
    options_ut.cpp
    text_options_ut.cpp
    all_losses_described.cpp
)

PEERDIR(
    catboost/private/libs/options
)


END()
