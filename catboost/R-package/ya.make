PYTEST()



TEST_CWD(catboost/R-package)

REQUIREMENTS(container:255627310)

SIZE(FAT)

TAG(
    ya:noretries
)

TEST_SRCS(
    test.py
)

DEPENDS(
    catboost/R-package/src
)

END()
