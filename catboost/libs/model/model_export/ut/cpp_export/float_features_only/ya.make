UNITTEST(model_export_cpp)



SIZE(MEDIUM)

PEERDIR(
    catboost/libs/model
    library/resource
)

DATA(
    arcadia/catboost/pytest/data/higgs/test_small
    arcadia/catboost/pytest/data/higgs/train_small
    arcadia/catboost/pytest/data/higgs/train.cd
)

RUN_PROGRAM(
    catboost/app fit
    -f ${ARCADIA_ROOT}/catboost/pytest/data/higgs/train_small
    --column-description ${ARCADIA_ROOT}/catboost/pytest/data/higgs/train.cd
    -i 100 -r 1234
    -m higgs_model --model-format CPP
    --model-format CatboostBinary
    --train-dir .
    CWD ${BINDIR}
    OUT higgs_model.cpp
    OUT_NOAUTO higgs_model.bin
    OUT_NOAUTO meta.tsv
)

RESOURCE(
    higgs_model.bin higgs_model_bin
)

SRCS(
    higgs_model.cpp
    test.cpp
)

DEPENDS(
    catboost/app
)

END()
