TOOL()



PEERDIR(
    library/resource
)

SRCS(
    main.cpp
)

INDUCED_DEPS(cpp library/resource/registry.h library/resource/resource.h)

END()
