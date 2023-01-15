

NEED_CHECK()

PY2_LIBRARY()

PY_SRCS(
    ymake_conf.py
)

PEERDIR(
    library/cpp/deprecated/enum_codegen
    library/cpp/deprecated/split
    library/cpp/string_utils/scan
)

END()

RECURSE(
    conf_fatal_error
    config
    docs/empty
    external_resources
    platform/java
    platform/perl
    platform/python
    platform/python/ldflags
    plugins
    prebuilt
    scripts
)
