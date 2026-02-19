#include <library/cpp/testing/unittest/registar.h>
#include <catboost/libs/metrics/metric.h>

Y_UNIT_TEST_SUITE(TTargetDependentQuantileMetricTest) {

    double q1 = 0.2;
    double q2 = 0.5;
    double q3 = 0.8;

    TVector<float> targets={1.,1.,3.,6.,7.,11.,12.};
    TVector<TVector<double>> approxes={{1.,3.,1.,7.,6.,12.,11.}};
    TVector<float> weights={1.,2.,3.,4.,5.,6.,7.};

    TVector<TString> loss_description = {"TargetDependentQuantile:boundaries=5,10;quantiles=0.2,0.5,0.8"};
    ui32 docCount = SafeIntegerCast<ui32>(targets.size());
    TVector<TQueryInfo> queryInfos;

    //integration test
    Y_UNIT_TEST(TargetDependentQuantileMetric1) {
        auto metric = CreateMetricsFromDescription(loss_description,1);
        NPar::TLocalExecutor localexec;
        auto result = dynamic_cast<TSingleTargetMetric*>(metric[0].Get())->Eval(approxes, targets, weights, queryInfos, 0, 1, localexec);
        UNIT_ASSERT_DOUBLES_EQUAL(result.Stats[0],0.,5e-6);
    }
    Y_UNIT_TEST(TargetDependentQuantileMetric2) {
        auto metric = CreateMetricsFromDescription(loss_description,1);
        NPar::TLocalExecutor localexec;
        auto result = dynamic_cast<TSingleTargetMetric*>(metric[0].Get())->Eval(approxes, targets, weights, queryInfos, 1, 2, localexec);
        UNIT_ASSERT_DOUBLES_EQUAL(result.Stats[0],-(targets[1]-approxes[0][1])*(1-q1)*weights[1],5e-6);
    }
    Y_UNIT_TEST(TargetDependentQuantileMetric3) {
        auto metric = CreateMetricsFromDescription(loss_description,1);
        NPar::TLocalExecutor localexec;
        auto result = dynamic_cast<TSingleTargetMetric*>(metric[0].Get())->Eval(approxes, targets, weights, queryInfos, 2, 3, localexec);
        UNIT_ASSERT_DOUBLES_EQUAL(result.Stats[0],(targets[2]-approxes[0][2])*q1*weights[2],5e-6);
    }
    Y_UNIT_TEST(TargetDependentQuantileMetric4) {
        auto metric = CreateMetricsFromDescription(loss_description,1);
        NPar::TLocalExecutor localexec;
        auto result = dynamic_cast<TSingleTargetMetric*>(metric[0].Get())->Eval(approxes, targets, weights, queryInfos, 3, 4, localexec);
        UNIT_ASSERT_DOUBLES_EQUAL(result.Stats[0],-(targets[3]-approxes[0][3])*(1-q2)*weights[3],5e-6);
    }
    Y_UNIT_TEST(TargetDependentQuantileMetric5) {
        auto metric = CreateMetricsFromDescription(loss_description,1);
        NPar::TLocalExecutor localexec;
        auto result = dynamic_cast<TSingleTargetMetric*>(metric[0].Get())->Eval(approxes, targets, weights, queryInfos, 4, 5, localexec);
        UNIT_ASSERT_DOUBLES_EQUAL(result.Stats[0],(targets[4]-approxes[0][4])*q2*weights[4],5e-6);
    }
    Y_UNIT_TEST(TargetDependentQuantileMetric6) {
        auto metric = CreateMetricsFromDescription(loss_description,1);
        NPar::TLocalExecutor localexec;
        auto result = dynamic_cast<TSingleTargetMetric*>(metric[0].Get())->Eval(approxes, targets, weights, queryInfos, 5, 6, localexec);
        UNIT_ASSERT_DOUBLES_EQUAL(result.Stats[0],-(targets[5]-approxes[0][5])*(1-q3)*weights[5],5e-6);
    }
    Y_UNIT_TEST(TargetDependentQuantileMetric7) {
        auto metric = CreateMetricsFromDescription(loss_description,1);
        NPar::TLocalExecutor localexec;
        auto result = dynamic_cast<TSingleTargetMetric*>(metric[0].Get())->Eval(approxes, targets, weights, queryInfos, 6, 7, localexec);
        UNIT_ASSERT_DOUBLES_EQUAL(result.Stats[0],(targets[4]-approxes[0][4])*q3*weights[6],1e-5);
    }

    //check that it all works together
    Y_UNIT_TEST(TargetDependentQuantileMetricAll) {
        auto metric = CreateMetricsFromDescription(loss_description,1);
        NPar::TLocalExecutor localexec;
        auto result = dynamic_cast<TSingleTargetMetric*>(metric[0].Get())->Eval(approxes, targets, weights, queryInfos, 1, docCount, localexec);
        UNIT_ASSERT_DOUBLES_EQUAL(result.Stats[0],-(targets[1]-approxes[0][1])*(1-q1)*weights[1]
                                                  +(targets[2]-approxes[0][2])*(q1)*weights[2]
                                                  -(targets[3]-approxes[0][3])*(1-q2)*weights[3]
                                                  +(targets[4]-approxes[0][4])*(q2)*weights[4]
                                                  -(targets[5]-approxes[0][5])*(1-q3)*weights[5]
                                                  +(targets[6]-approxes[0][6])*(q3)*weights[6]
        ,1e-4);
    }

}
