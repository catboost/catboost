IF (NOT OS_WINDOWS)
    UNITTEST(model_export_cpp)

    

    SIZE(MEDIUM)

    PEERDIR(
        catboost/libs/model
        library/resource
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
        -m adult_model --model-format CPP
        --model-format CatboostBinary
        --train-dir .
        CWD ${BINDIR}
        OUT adult_model.cpp
        OUT_NOAUTO adult_model.bin
        OUT_NOAUTO meta.tsv
    )

    RESOURCE(
        adult_model.bin adult_model_bin
    )

    SRCS(
        adult_model.cpp
        test.cpp
    )

    DEPENDS(
        catboost/app
    )
    END()
ENDIF()
