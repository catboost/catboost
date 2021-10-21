

PROGRAM()

LICENSE(Apache-2.0)

LICENSE_TEXTS(.yandex_meta/licenses.list.txt)

NO_UTIL()

SRCS(
    monster.fbs
    sample_binary.cpp
)

PEERDIR(
    contrib/libs/flatbuffers
)

END()
