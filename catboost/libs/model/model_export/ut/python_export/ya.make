PYTEST()



FORK_TESTS()
FORK_SUBTESTS()

SIZE(SMALL)

PEERDIR(
    catboost/python-package/lib
    catboost/pytest/lib
    contrib/python/numpy
    library/python/resource
)

DATA(
    arcadia/catboost/pytest/data/adult/test_small
    arcadia/catboost/pytest/data/adult/train_small
    arcadia/catboost/pytest/data/adult/train.cd
    arcadia/catboost/pytest/data/higgs/test_small
    arcadia/catboost/pytest/data/higgs/train_small
    arcadia/catboost/pytest/data/higgs/train.cd
)

RUN_PROGRAM(
    catboost/app fit
    -f ${ARCADIA_ROOT}/catboost/pytest/data/adult/train_small
    --column-description ${ARCADIA_ROOT}/catboost/pytest/data/adult/train.cd
    -i 100 -r 1234
    -m adult_model --model-format Python
    --model-format CatboostBinary
    --train-dir .
    CWD ${BINDIR}
    OUT adult_model.py
    OUT_NOAUTO adult_model.bin
    OUT_NOAUTO meta.tsv
)

RESOURCE(
    adult_model.bin cb_adult_model_bin
)

TEST_SRCS(
    adult_model.py
    test.py
)

DEPENDS(
    catboost/app
)

END()
