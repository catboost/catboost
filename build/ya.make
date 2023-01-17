

NEED_CHECK()

PY2_LIBRARY()

PY_SRCS(ymake_conf.py)

PEERDIR(
    library/cpp/deprecated/enum_codegen
    library/cpp/deprecated/split
    library/cpp/string_utils/scan
    library/cpp/deprecated/atomic
)

END()

RECURSE(
    conf_fatal_error
    config
    docs/empty
    external_resources
    platform/java
    platform/local_so
    platform/perl
    platform/python
    platform/python/ldflags
    platform/
    plugins
    prebuilt
    scripts
    sysincl/check
)
