PROGRAM()



PEERDIR(
    library/cpp/resource
)

SRCS(
    main.cpp
)

INDUCED_DEPS(cpp ${ARCADIA_ROOT}/library/resource/registry.h)

END()
