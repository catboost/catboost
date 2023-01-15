

LIBRARY()

SRCS(
   countdown_latch.cpp
   spin_wait.cpp
   helpers.cpp
)

PEERDIR(
    catboost/libs/helpers
)

END()
