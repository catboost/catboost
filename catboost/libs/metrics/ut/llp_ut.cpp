#include <library/cpp/testing/unittest/registar.h>

#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/metrics/metric_holder.h>

#include <util/generic/array_ref.h>


static TVector<TVector<double>> GenerateApproxesFromProbs(TConstArrayRef<double> probs) {
    TVector<double> approx(probs.size());
    for (int i = 0; i < approx.ysize(); i++) {
        approx[i] = log(probs[i]) - log(1. - probs[i]);
    }
    return {approx};
}

static TVector<TVector<double>> GenerateTestApproxes() {
    return GenerateApproxesFromProbs({0.9, 0.1, 0.1});
}

Y_UNIT_TEST_SUITE(LLPMetricTest) {
Y_UNIT_TEST(LLPTest) {
    {
        TVector<TVector<double>> approx = GenerateTestApproxes();
        TVector<float> target{1, 0, 0};

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::LogLikelihoodOfPrediction, TLossParams(),
                                      /*approxDimension=*/1)[0]);
        metric->UseWeights = true;
        TMetricHolder score = metric->Eval(approx, target, {}, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 1.59346f, 1e-5);
    }
    {
        TVector<TVector<double>> approx = GenerateTestApproxes();
        TVector<float> target{0, 0, 0};
        TVector<float> weight{1, 1, 1};

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::LogLikelihoodOfPrediction, TLossParams(),
                                      /*approxDimension=*/1)[0]);
        metric->UseWeights = true;
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_EQUAL(metric->GetFinalError(score), 0);
    }
    {
        TVector<TVector<double>> approx = GenerateTestApproxes();
        TVector<float> target{1, 1, 1};
        TVector<float> weight{1, 1, 1};

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::LogLikelihoodOfPrediction, TLossParams(),
                                      /*approxDimension=*/1)[0]);
        metric->UseWeights = true;
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_EQUAL(metric->GetFinalError(score), 0);
    }
    {
        TVector<TVector<double>> approx = GenerateTestApproxes();
        TVector<float> target{1, 0, 0};
        TVector<float> weight{1, 1, 1};

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::LogLikelihoodOfPrediction, TLossParams(),
                                      /*approxDimension=*/1)[0]);
        metric->UseWeights = true;

        for (int i = 1; i <= 10; i++) {
            Fill(weight.begin(), weight.end(), i / 3.);
            TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);
            UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 1.5934609f, 1e-5);
        }
    }
    {
        TVector<TVector<double>> approx = GenerateApproxesFromProbs({0.9, 0.8, 0.1, 0.3, 0.1});
        TVector<float> target{0.3f, 0.99f, 0.1f, 0.3f, 0};
        TVector<float> weight{0.9f, 0.25f, 9.f, 20.f, 1.f};

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::LogLikelihoodOfPrediction, TLossParams(),
                                      /*approxDimension=*/1)[0]);
        metric->UseWeights = true;
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.0420202789f, 1e-5);
    }
    {
        TVector<TVector<double>> approx = GenerateApproxesFromProbs({0.9, 0.7, 0.1});
        TVector<float> target{1, 1, 0};
        TVector<float> weight{10.f, 10.f, 90.f};

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::LogLikelihoodOfPrediction, TLossParams(),
                                      /*approxDimension=*/1)[0]);
        metric->UseWeights = true;
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 1.90262617f, 1e-5);
    }
    {
        TVector<TVector<double>> approx = GenerateApproxesFromProbs({0.9, 0.7, 0.1});
        TVector<float> target{1, 0, 0};
        TVector<float> weight{90.f, 10.f, 10.f};

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::LogLikelihoodOfPrediction, TLossParams(),
                                      /*approxDimension=*/1)[0]);
        metric->UseWeights = true;
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0.328661609f, 1e-5);
    }
}
}
