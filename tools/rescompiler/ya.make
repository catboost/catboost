PROGRAM()



PEERDIR(
    library/cpp/resource
)

SRCS(
    main.cpp
)

INDUCED_DEPS(cpp ${ARCADIA_ROOT}/library/cpp/resource/registry.h ${ARCADIA_ROOT}/library/cpp/resource/resource.h)

END()
