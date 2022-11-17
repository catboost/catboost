PY23_LIBRARY()



PY_SRCS(
    __init__.py
    base.py
    library.py
    webpack_bundle.py
)

PEERDIR(
    build/plugins/lib/nots/package_manager
    build/plugins/lib/nots/typescript
)

END()
