LIBRARY()




SRCS(
   countdown_latch.cpp
   cpu_random.cpp
   compression_helpers.cpp
   spin_wait.cpp
)

PEERDIR(
    catboost/libs/helpers
)



END()
