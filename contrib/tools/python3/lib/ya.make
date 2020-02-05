LIBRARY()



LICENSE(Python-2.0)

PEERDIR(
    contrib/tools/python3/src
    contrib/tools/python3/src/Lib
    contrib/tools/python3/src/Modules
)

SUPPRESSIONS(lsan.supp)

END()
