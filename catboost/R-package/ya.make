PYTEST()



TEST_CWD(catboost/R-package)

REQUIREMENTS(container:575178824)

SIZE(LARGE)

TAG(
    ya:noretries
    ya:fat
    ya:force_sandbox
)

TEST_SRCS(
    test.py
)

DEPENDS(
    catboost/R-package/src
)

END()

RECURSE(
    src
)
