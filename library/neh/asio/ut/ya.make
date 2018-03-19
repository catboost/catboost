UNITTEST()



PEERDIR(
    ADDINCL library/neh/asio
)

SRCDIR(library/neh/asio)

SRCS(
    asio_ut.cpp
    io_service_impl_ut.cpp
)

END()
