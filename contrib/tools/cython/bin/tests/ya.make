PY3TEST()

WITHOUT_LICENSE_TEXTS()

FORK_SUBTESTS()

SPLIT_FACTOR(4)

DEPENDS(
    contrib/tools/cython/bin
)

DATA(
    arcadia/build/conf/sysincl.conf
    arcadia/build/sysincl
    arcadia/contrib/python/numpy/py3/numpy/__init__.pxd
    arcadia/contrib/tools/cython
)

TEST_SRCS(
    test_compiler.py
    test_recipe.py
)

END()
