PY23_LIBRARY()

VERSION(1.8.1)

LICENSE(
    MIT
)



NO_CHECK_IMPORTS(
    py._code._assertionnew
    py._code._assertionold
)

PY_SRCS(
    TOP_LEVEL
    py/_builtin.py
    py/_code/_assertionnew.py
    py/_code/_assertionold.py
    py/_code/assertion.py
    py/_code/code.py
    py/_code/__init__.py
    py/_code/_py2traceback.py
    py/_code/source.py
    py/_error.py
    py/__init__.py
    py/_io/capture.py
    py/_io/__init__.py
    py/_io/saferepr.py
    py/_io/terminalwriter.py
    py/_log/__init__.py
    py/_log/log.py
    py/_log/warning.py
    py/__metainfo.py
    py/_path/cacheutil.py
    py/_path/common.py
    py/_path/__init__.py
    py/_path/local.py
    py/_path/svnurl.py
    py/_path/svnwc.py
    py/_process/cmdexec.py
    py/_process/forkedfunc.py
    py/_process/__init__.py
    py/_process/killproc.py
    py/_std.py
    py/test.py
    py/_vendored_packages/apipkg.py
    py/_vendored_packages/iniconfig.py
    py/_vendored_packages/__init__.py
    py/_version.py
    py/_xmlgen.py
)

RESOURCE_FILES(
    PREFIX contrib/python/py/
    .dist-info/METADATA
    .dist-info/top_level.txt
)

PEERDIR(
    contrib/python/six
)

NO_LINT()

END()
