#include <library/cpp/testing/unittest/registar.h>

#include <catboost/libs/metrics/optimal_const_for_loss.h>
#include <catboost/private/libs/options/restrictions.h>
#include <catboost/libs/metrics/metric.h>

#include <util/generic/array_ref.h>
#include <util/generic/fwd.h>
#include <util/generic/vector.h>

#include <cfloat>

using namespace NCB;

Y_UNIT_TEST_SUITE(MultiQuantileTest) {

    const TVector<float> unitWeights = {1, 1, 1, 1, 1, 1, 1, 1};
    const TVector<float> target = {0, 2, 10, 37, 40, 500, 501, 600};

    Y_UNIT_TEST(CreateAndCalc) {


        const TVector<double> alpha = {0.25, 0.5, 0.75};
        const auto params = TLossParams::FromVector(
            {
                {ToString("alpha"), ToString(alpha[0]) + "," + ToString(alpha[1]) + "," + ToString(alpha[2])}
            });

        const auto multiQuantile = CreateMetric(ELossFunction::MultiQuantile, params, alpha.size());
        UNIT_ASSERT_EQUAL(multiQuantile.size(), 1);

        TVector<TVector<double>> approx(
            {
                {0, 2, 10, 37, 40, 500, 501, 600},
                {0, 2, 10, 37, 40, 500, 501, 600},
                {0, 2, 10, 37, 40, 500, 501, 600}
            });
        NPar::TLocalExecutor localExecutor;
        const auto metric = dynamic_cast<const ISingleTargetEval&>(*multiQuantile[0]).Eval(
            approx,
            target,
            unitWeights,
            /*queriesInfo*/{},
            /*begin*/0,
            /*end*/approx.size(),
            localExecutor
        );
    }
}
