#include "model_test_helpers.h"

#include <catboost/libs/data/load_data.h>
#include <catboost/libs/train_lib/train_model.h>

TFullModel TrainFloatCatboostModel() {
    TPool pool;
    pool.Docs.Resize(/*doc count*/3, /*factors count*/ 3, /*baseline dimension*/ 0, /*has queryId*/ false, /*has subgroupId*/ false);
    pool.Docs.Factors[0] = {+0.5f, +1.5f, -2.5f};
    pool.Docs.Factors[1] = {+0.7f, +6.4f, +2.4f};
    pool.Docs.Factors[2] = {-2.0f, -1.0f, +6.0f};
    pool.Docs.Target = {1.0f, 0.0f, 0.2f};

    TFullModel model;
    TEvalResult evalResult;
    NJson::TJsonValue params;
    params.InsertValue("iterations", 5);
    TrainModel(
            params,
            Nothing(),
            Nothing(),
            TClearablePoolPtrs(pool, {&pool}),
            "",
            &model,
            {&evalResult}
    );

    return model;
}

TPool GetAdultPool() {
    TOFStream poolOutput("pool.txt");
    poolOutput <<
        "0\t1\tn\t1\t57.0\tSelf-emp-not-inc\t174760.0\tAssoc-acdm\t12.0\tMarried-spouse-absent\tFarming-fishing\tUnmarried\tAmer-Indian-Eskimo\tMale\t0.0\t0.0\t40.0\tUnited-States\n"
        "0\t1\tn\t1\t26.0\tPrivate\t94392.0\t11th\t7.0\tSeparated\tOther-service\tUnmarried\tWhite\tFemale\t0.0\t0.0\t20.0\tUnited-States\n"
        "0\t1\tn\t1\t55.0\tPrivate\t451603.0\tHS-grad\t9.0\tDivorced\tCraft-repair\tUnmarried\tWhite\tMale\t0.0\t0.0\t40.0\tUnited-States\n"
        "0\t1\tn\t1\t52.0\tPrivate\t84788.0\t10th\t6.0\tNever-married\tMachine-op-inspct\tNot-in-family\tWhite\tMale\t0.0\t0.0\t40.0\tUnited-States\n"
        "0\t1\tn\t1\t23.0\tPrivate\t204641.0\t10th\t6.0\tNever-married\tHandlers-cleaners\tUnmarried\tWhite\tMale\t0.0\t0.0\t50.0\tUnited-States\n"
        "0\t1\tn\t1\t44.0\t?\t210875.0\t11th\t7.0\tDivorced\t?\tNot-in-family\tBlack\tFemale\t0.0\t0.0\t40.0\tUnited-States\n"
        "0\t0\tn\t1\t51.0\tLocal-gov\t117496.0\tMasters\t14.0\tMarried-civ-spouse\tProf-specialty\tWife\tWhite\tFemale\t0.0\t0.0\t40.0\tUnited-States\n"
        "0\t0\tn\t1\t44.0\tPrivate\t238574.0\tProf-school\t15.0\tMarried-civ-spouse\tProf-specialty\tHusband\tWhite\tMale\t0.0\t0.0\t50.0\tUnited-States\n"
        "0\t0\tn\t1\t62.0\t?\t191118.0\tSome-college\t10.0\tMarried-civ-spouse\t?\tHusband\tWhite\tMale\t7298.0\t0.0\t40.0\tUnited-States\n"
        "0\t0\tn\t1\t46.0\tSelf-emp-not-inc\t198759.0\tProf-school\t15.0\tMarried-civ-spouse\tProf-specialty\tHusband\tWhite\tMale\t0.0\t2415.0\t80.0\tUnited-States\n"
        "0\t0\tn\t1\t37.0\tPrivate\t215503.0\tHS-grad\t9.0\tMarried-civ-spouse\tExec-managerial\tHusband\tWhite\tMale\t4386.0\t0.0\t45.0\tUnited-States\n"
        "0\t0\tn\t1\t49.0\tSelf-emp-inc\t158685.0\tHS-grad\t9.0\tMarried-civ-spouse\tAdm-clerical\tWife\tWhite\tFemale\t0.0\t2377.0\t40.0\tUnited-States\n"
        "0\t0\tn\t1\t52.0\tLocal-gov\t311569.0\tMasters\t14.0\tMarried-civ-spouse\tExec-managerial\tHusband\tWhite\tMale\t0.0\t0.0\t50.0\tUnited-States" << Endl;
    TOFStream cdOutput("pool.txt.cd");
    cdOutput << "1\tTarget" << Endl;
    TVector<int> categIdx = {0, 2, 3, 5, 7, 9, 10, 11, 12, 13, 17};
    for (int i : categIdx) {
        cdOutput << i << "\tCateg" << Endl;
    }
    TPool pool;
    NCatboostOptions::TDsvPoolFormatParams dsvPoolFormatParams;
    dsvPoolFormatParams.CdFilePath = NCB::TPathWithScheme("dsv://pool.txt.cd");
    NCB::ReadPool(
            NCB::TPathWithScheme("dsv://pool.txt"),
            NCB::TPathWithScheme(),
            NCB::TPathWithScheme(),
            dsvPoolFormatParams,
            {},
            16,
            true,
            &pool);
    return pool;
}
