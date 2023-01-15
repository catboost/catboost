PY23_LIBRARY()



PY_SRCS(
    core_proc.py
)

RESOURCE(
    library/python/coredump_filter/core_proc.js /coredump_filter_data/core_proc.js
    library/python/coredump_filter/epilog.html /coredump_filter_data/epilog.html
    library/python/coredump_filter/prolog.html /coredump_filter_data/prolog.html
    library/python/coredump_filter/styles.css /coredump_filter_data/styles.css
)

PEERDIR(
    contrib/libs/jquery
)

END()

RECURSE(
    minidump2core
    python_tracer
)
