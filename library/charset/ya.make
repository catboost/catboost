LIBRARY()



SRCS(
    generated/cp_data.cpp
    generated/encrec_data.cpp
    codepage.cpp
    cp_encrec.cpp
    doccodes.cpp
    iconv.cpp
    recyr.hh
    recyr_int.hh
    ci_string.cpp
    wide.cpp
)

PEERDIR(
    contrib/libs/libiconv
)

END()
