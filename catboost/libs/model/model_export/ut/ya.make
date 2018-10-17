IF (${TARGET_PLATFORM} STREQUAL ${HOST_PLATFORM})

    PYTEST()

    

    SIZE(MEDIUM)

    TEST_SRCS(
        test.py
    )

    PEERDIR(
        catboost/pytest/lib
        catboost/python-package/lib
    )

    DEPENDS(
        catboost/app
        catboost/tools/limited_precision_dsv_diff
    )

    DATA(
        arcadia/catboost/libs/model/model_export/ut/applicator.cpp
        arcadia/catboost/pytest/data/adult/test_small
        arcadia/catboost/pytest/data/adult/train_small
        arcadia/catboost/pytest/data/adult/train.cd
        arcadia/catboost/pytest/data/higgs/test_small
        arcadia/catboost/pytest/data/higgs/train_small
        arcadia/catboost/pytest/data/higgs/train.cd
    )

    END()

ENDIF()
