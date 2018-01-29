

PACKAGE()

PEERDIR(
    library/deprecated/enum_codegen
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
)
