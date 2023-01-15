PY23_LIBRARY(ipython_genutils)

LICENSE(BSD-3-Clause)

VERSION(0.2.0)



PY_SRCS(
    TOP_LEVEL
    ipython_genutils/__init__.py
    ipython_genutils/_version.py
    ipython_genutils/encoding.py
    ipython_genutils/importstring.py
    ipython_genutils/ipstruct.py
    ipython_genutils/path.py
    ipython_genutils/py3compat.py
    ipython_genutils/tempdir.py
    ipython_genutils/testing/__init__.py
    ipython_genutils/testing/decorators.py
    ipython_genutils/text.py
)

NO_LINT()

END()
