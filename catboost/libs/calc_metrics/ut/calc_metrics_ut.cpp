#include <catboost/libs/calc_metrics/calc_metrics.h>

#include <catboost/libs/data/load_data.h>
#include <catboost/libs/eval_result/eval_result.h>
#include <catboost/libs/model/ut/lib/model_test_helpers.h>
#include <catboost/libs/train_lib/train_model.h>
#include <catboost/private/libs/algo/apply.h>

#include <library/cpp/testing/unittest/registar.h>
#include <utility>
#include <util/string/builder.h>


using namespace std;
using namespace NCB;


void BuildApproxAsPoolFileAndGetMetric(
    NPar::TLocalExecutor* executor,
    THashMap<TString, double>* metricsResultPtr,
    TPathWithScheme* dstPathRes,
    THashMap<int, EColumn>* nonAuxiliaryColumnsDescriptionPtr) {
    NJson::TJsonValue params;
    params.InsertValue("loss_function", "Logloss");
    params.InsertValue("custom_metric", NJson::TJsonArray({"Accuracy", "AUC"}));
    params.InsertValue("learning_rate", 0.3);
    params.InsertValue("iterations", 7);
    TFullModel model;
    TEvalResult evalResult;

    TDataProviderPtr pool = GetAdultPool();

    TMetricsAndTimeLeftHistory metricsAndTimeHistory;

    TrainModel(
        params,
        nullptr,
        Nothing(),
        Nothing(),
        Nothing(),
        TDataProviders{pool, {pool}},
        /*initModel*/ Nothing(),
        /*initLearnProgress*/ nullptr,
        "",
        &model,
        {&evalResult},
        &metricsAndTimeHistory);

    (*metricsResultPtr) = std::move(metricsAndTimeHistory.TestMetricsHistory.back()[0]);

    const TExternalLabelsHelper visibleLabelsHelper(model);

    auto tmpFileName = MakeTempName();
    TFileOutput output(tmpFileName);
    *dstPathRes = TPathWithScheme(tmpFileName, "dsv");

    TOFStream fileStream(tmpFileName);

    OutputEvalResultToFile(
        evalResult,
        executor,
        /* outputColumns */ {"Label", "RawFormulaVal"},
        model.GetLossFunctionName(),
        visibleLabelsHelper,
        *pool,
        &fileStream,
        *dstPathRes,
        {0, 1},
        NCB::TDsvFormatOptions(),
        /*writeHeader*/ false);

    nonAuxiliaryColumnsDescriptionPtr->insert({0, EColumn::Label});
    nonAuxiliaryColumnsDescriptionPtr->insert({1, EColumn::Num});

}

TDataProviderPtr BuildApproxAsPoolAndGetMetric(NPar::TLocalExecutor* executor, THashMap<TString, double>* metricsResultPtr) {
    TPathWithScheme dstPath;
    THashMap<int, EColumn> nonAuxiliaryColumnsDescription;
    BuildApproxAsPoolFileAndGetMetric(executor, metricsResultPtr, &dstPath, &nonAuxiliaryColumnsDescription);

    TStringBuilder cdOutputData;
    for (const auto& [key, value] : nonAuxiliaryColumnsDescription) {
        cdOutputData << key << "\t" << value << Endl;
    }
    auto cdTmpFileName = MakeTempName();
    {
        TFileOutput cdOutput(cdTmpFileName);
        cdOutput.Write(cdOutputData);
        cdOutput.Write("\n");
    }

    NCatboostOptions::TColumnarPoolFormatParams columnarPoolFormatParams;
    columnarPoolFormatParams.CdFilePath = TPathWithScheme(cdTmpFileName);

    TVector<NJson::TJsonValue> classLabels;

    return ReadDataset(
        /*taskType*/Nothing(),
                    dstPath,
                    TPathWithScheme(),
                    TPathWithScheme(),
                    TPathWithScheme(),
                    TPathWithScheme(),
                    TPathWithScheme(),
                    TPathWithScheme(),
                    TPathWithScheme(),
                    columnarPoolFormatParams,
        /*ignoredFeatures*/ {},
                    EObjectsOrder::Undefined,
        /*threadCount*/ 16,
        /*verbose*/true,
        /*loadSampleIds*/ false,
        /*forceUnitAutoPairWeights*/false,
                    &classLabels
    );
}

Y_UNIT_TEST_SUITE(TCalcMetrics) {
    Y_UNIT_TEST(TestCalcMetricsPoolConsumer) {

        NPar::TLocalExecutor executor;

        THashMap<TString, double> trainMetricsResult;
        auto approxPool = BuildApproxAsPoolAndGetMetric(&executor, &trainMetricsResult);
        TVector<TString> metricName = {"Accuracy", "AUC"};
        TVector<THolder<IMetric>> metrics = CreateMetricsFromDescription(metricName, 1);

        TNonAdditiveMetricData nonAdditiveMetricData;
        TVector<const IMetric*> additiveMetrics;
        TVector<const IMetric*> nonAdditiveMetrics;

        for (const auto& metric : metrics) {
            if (metric->IsAdditiveMetric()) {
                additiveMetrics.push_back(metric.Get());
            } else {
                nonAdditiveMetrics.push_back(metric.Get());
            }
        }

        TVector<TMetricHolder> stats = ConsumeCalcMetricsData(
            additiveMetrics,
            approxPool,
            &executor);
        nonAdditiveMetricData.SaveProcessedData(approxPool, &executor);

        auto nonAdditiveStats = CalculateNonAdditiveMetrics(
            nonAdditiveMetricData,
            nonAdditiveMetrics,
            &executor
        );
        stats.insert(
            stats.end(),
            nonAdditiveStats.begin(),
            nonAdditiveStats.end()
        );

        TVector<double> metricResults = GetMetricResultsFromMetricHolder(stats, metrics);

        Y_ASSERT(metricName.size() == metrics.size());
        for (auto ind : xrange(metricName.size())) {
            UNIT_ASSERT_DOUBLES_EQUAL(metricResults[ind], trainMetricsResult.at(metricName[ind]), 1e-7);
        }
    }

    Y_UNIT_TEST(TestCalcMetricsSingleHost) {

        NPar::TLocalExecutor executor;

        THashMap<TString, double> trainMetricsResult;
        TPathWithScheme inputPath;
        THashMap<int, EColumn> nonAuxiliaryColumnsDescription;
        BuildApproxAsPoolFileAndGetMetric(&executor, &trainMetricsResult, &inputPath, &nonAuxiliaryColumnsDescription);

        TVector<TString> metricName = {"Logloss", "Accuracy", "AUC"};
        TVector<THolder<IMetric>> metrics = CreateMetricsFromDescription(metricName, 1);

        auto metricResults = TOpenSourceCalcMetricsImplementation().CalcMetrics(
            inputPath,
            metricName,
            NCB::TDsvFormatOptions(),
            nonAuxiliaryColumnsDescription,
            /* threadCount */ NSystemInfo::CachedNumberOfCpus() - 1);

        Y_ASSERT(metricName.size() == metrics.size());
        for (auto ind : xrange(metricName.size())) {
            UNIT_ASSERT_DOUBLES_EQUAL(metricResults.at(metricName[ind]), trainMetricsResult.at(metricName[ind]), 1e-7);
        }
    }
}
