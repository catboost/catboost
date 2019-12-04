

NEED_CHECK()

PACKAGE()

PEERDIR(
    library/deprecated/enum_codegen
    library/deprecated/split
    library/string_utils/scan
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
    conf_fatal_error
    plugins
    scripts
    platform/perl
    platform/perl/5.14.4
    platform/python
    platform/python/ldflags
    platform/java
)
