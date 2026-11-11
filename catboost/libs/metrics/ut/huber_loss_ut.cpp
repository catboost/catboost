#include <library/cpp/testing/unittest/registar.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/metrics/metric_holder.h>

Y_UNIT_TEST_SUITE(HuberLossMetricTest) {
Y_UNIT_TEST(HuberLossTest) {
    {
        TVector<TVector<double>> approx{{0, 0, 0, 0}};
        TVector<float> target{0, 0, 0, 0};
        TVector<float> weight;

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::Huber, TLossParams::FromVector({{"delta", "1.0"}}), /*approxDimension=*/1)[0]);
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 0, 1e-6);
    }
    {
        TVector<TVector<double>> approx{{0, 0, 0, 0}};
        TVector<float> target{1, 2, 3, 4};
        TVector<float> weight{0.26705f, 0.666578f, 0.6702279f, 0.3976618f};

        NPar::TLocalExecutor executor;
        const auto metric = std::move(CreateSingleTargetMetric(ELossFunction::Huber, TLossParams::FromVector({{"delta", "1.0"}}), /*approxDimension=*/1)[0]);
        metric->UseWeights = true;
        TMetricHolder score = metric->Eval(approx, target, weight, {}, 0, target.size(), executor);

        UNIT_ASSERT_DOUBLES_EQUAL(metric->GetFinalError(score), 2.0987963, 1e-6);
    }
}
}

Y_UNIT_TEST_SUITE(CombinationMetricDescriptionTest) {
Y_UNIT_TEST(ComponentParametersAndWeightsRoundTrip) {
    const TString description =
        "Combination:loss0=RMSE;weight0=2.0;loss1=Huber:delta=1.5;weight1=0.6;"
        "loss2=QueryRMSE;weight2=0.25";
    const auto metrics = CreateMetricsFromDescription({description}, 1);
    UNIT_ASSERT_VALUES_EQUAL(metrics.size(), 1);
    const auto formatted = metrics[0]->GetDescription();
    UNIT_ASSERT_VALUES_EQUAL(formatted, description);
    UNIT_ASSERT_VALUES_EQUAL(ParseLossType(formatted), ELossFunction::Combination);
    const auto original = NCatboostOptions::ParseLossDescription(description);
    const auto roundTrip = NCatboostOptions::ParseLossDescription(formatted);
    UNIT_ASSERT(original.GetLossParamsMap() == roundTrip.GetLossParamsMap());
    const auto recreated = CreateMetricFromDescription(roundTrip, 1);
    UNIT_ASSERT_VALUES_EQUAL(recreated[0]->GetDescription(), formatted);
}

Y_UNIT_TEST(WeightedEvalObjectiveAndCustomMetricsKeepDistinctDescriptions) {
    const auto objective = NCatboostOptions::ParseLossDescription(
        "Combination:loss0=RMSE;weight0=2.0;loss1=Huber:delta=1.5;weight1=0.6");
    const auto evaluation = NCatboostOptions::ParseLossDescription(
        "Combination:loss0=RMSE;weight0=0.25;loss1=Huber:delta=0.8;weight1=1.5");
    NCatboostOptions::TOption<NCatboostOptions::TMetricOptions> options(
        "metrics", NCatboostOptions::TMetricOptions());
    options->ObjectiveMetric.Set(objective);
    options->EvalMetric.Set(evaluation);
    options->CustomMetrics.Set(TVector<NCatboostOptions::TLossDescription>{objective, evaluation});
    // This path calls ShouldConsiderWeightsByDefault, which parses the metric
    // description before the Metal progress adapter is constructed.
    const auto metrics = CreateMetrics(options, Nothing(), 1, true);
    UNIT_ASSERT_VALUES_EQUAL(metrics.size(), 6);
    UNIT_ASSERT_VALUES_EQUAL(
        NCatboostOptions::ParseLossDescription(metrics[0]->GetDescription()).GetLossParamsMap().at("weight0"),
        "0.25");
    UNIT_ASSERT_VALUES_EQUAL(
        NCatboostOptions::ParseLossDescription(metrics[1]->GetDescription()).GetLossParamsMap().at("weight0"),
        "2.0");
    for (size_t index = 0; index < metrics.size(); ++index) {
        UNIT_ASSERT_VALUES_EQUAL(ParseLossType(metrics[index]->GetDescription()), ELossFunction::Combination);
        for (size_t previous = 0; previous < index; ++previous) {
            UNIT_ASSERT(metrics[index]->GetDescription() != metrics[previous]->GetDescription());
        }
    }
    UNIT_ASSERT_VALUES_EQUAL(metrics[2]->UseWeights.Get(), true);
    UNIT_ASSERT_VALUES_EQUAL(metrics[3]->UseWeights.Get(), false);
    UNIT_ASSERT_VALUES_EQUAL(metrics[4]->UseWeights.Get(), true);
    UNIT_ASSERT_VALUES_EQUAL(metrics[5]->UseWeights.Get(), false);
}
}
