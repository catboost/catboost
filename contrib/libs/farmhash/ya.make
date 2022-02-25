LIBRARY()

LICENSE(MIT)

LICENSE_TEXTS(.yandex_meta/licenses.list.txt)

VERSION(2017-06-26-23eecfbe7e84ebf2e229bd02248f431c36e12f1a)



ADDINCL(GLOBAL contrib/libs/farmhash/include)

PEERDIR(
    contrib/libs/farmhash/arch/sse41
    contrib/libs/farmhash/arch/sse42
    contrib/libs/farmhash/arch/sse42_aesni
)

NO_COMPILER_WARNINGS()

SRCS(
    farmhashuo.cc
    farmhashxo.cc
    farmhashna.cc
    farmhashmk.cc
    farmhashcc.cc
    farmhash_iface.cc
)

END()
