

NEED_CHECK()

PACKAGE()

PEERDIR(
    library/deprecated/enum_codegen
    library/deprecated/split
)

END()

ADD_TEST(
    PEP8
        ymake_conf.py
)

ADD_TEST(
    PY_FLAKES
        ymake_conf.py
)

RECURSE(
    plugins
    scripts
    platform/perl
    platform/perl/5.14.4
    platform/python
    platform/python/ldflags
    platform/java
)
