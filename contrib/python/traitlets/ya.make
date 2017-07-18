LIBRARY()

# Version: 4.3.1



PEERDIR(
    contrib/python/decorator-4.0.6
    contrib/python/enum34
    contrib/python/ipython_genutils-0.1.0
    contrib/python/six
)

PY_SRCS(
    TOP_LEVEL
    traitlets/__init__.py
    traitlets/_version.py
    traitlets/config/__init__.py
    traitlets/config/application.py
    traitlets/config/configurable.py
    traitlets/config/loader.py
    traitlets/config/manager.py
    traitlets/log.py
    traitlets/traitlets.py
    traitlets/utils/__init__.py
    traitlets/utils/bunch.py
    traitlets/utils/getargspec.py
    traitlets/utils/importstring.py
    traitlets/utils/sentinel.py
)

NO_LINT()

END()
