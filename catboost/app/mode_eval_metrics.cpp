#include "modes.h"

#include <catboost/private/libs/algo/plot.h>
#include <catboost/libs/data/proceed_pool_in_blocks.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/mem_usage.h>
#include <catboost/private/libs/labels/label_converter.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/model/model.h>
#include <catboost/private/libs/options/analytical_mode_params.h>
#include <catboost/private/libs/options/loss_description.h>
#include <catboost/private/libs/target/data_providers.h>

#include <library/getopt/small/last_getopt_opts.h>
#include <library/getopt/small/last_getopt_parse_result.h>

#include <util/folder/tempdir.h>
#include <util/string/split.h>
#include <util/system/compiler.h>


using namespace NCB;


struct TModeEvalMetricsParams {
    ui32 Step = 1;
    ui32 FirstIteration = 0;
    ui32 EndIteration = 0;
    int ReadBlockSize;
    TString MetricsDescription;
    TString ResultDirectory;
    TString TmpDir;

    void BindParserOpts(NLastGetopt::TOpts& parser) {
        parser.AddLongOption("ntree-start", "Start iteration.")
                .RequiredArgument("INT")
                .StoreResult(&FirstIteration);
        parser.AddLongOption("ntree-end", "End iteration.")
                .RequiredArgument("INT")
                .StoreResult(&EndIteration);
        parser.AddLongOption("eval-period", "Eval metrics every eval-period trees.")
                .RequiredArgument("INT")
                .StoreResult(&Step);
        parser.AddLongOption("metrics", "coma-separated eval metrics")
                .RequiredArgument("String")
                .StoreResult(&MetricsDescription);
        parser.AddLongOption("result-dir", "directory with results")
                .RequiredArgument("String")
                .StoreResult(&ResultDirectory);
        parser.AddLongOption("block-size", "Compute block size")
                .RequiredArgument("INT")
                .DefaultValue("150000")
                .StoreResult(&ReadBlockSize);
        parser.AddLongOption("tmp-dir", "Dir to store approx for non-additive metrics. Use \"-\" to generate directory.")
                .RequiredArgument("String")
                .DefaultValue("-")
                .StoreResult(&TmpDir);
    }
};


static TVector<NCatboostOptions::TLossDescription> CreateMetricDescriptions(
    TStringBuf metricsDescription) {

    TVector<NCatboostOptions::TLossDescription> result;
    for (const auto& metricDescription : StringSplitter(metricsDescription).Split(',').SkipEmpty()) {
        result.emplace_back(NCatboostOptions::ParseLossDescription(metricDescription.Token()));
    }
    CB_ENSURE(!result.empty(), "No metric in metrics description " << metricsDescription);

    return result;
}


static void ReadDatasetParts(
    const NCB::TAnalyticalModeCommonParams& params,
    int blockSize,
    TConstArrayRef<NCatboostOptions::TLossDescription> metricDescriptions,
    const TFullModel& model,
    TRestorableFastRng64* rand,
    NPar::TLocalExecutor* executor,
    TVector<TProcessedDataProvider>* processedDatasetParts) {

    processedDatasetParts->clear();
    ReadAndProceedPoolInBlocks(params.DatasetReadingParams, blockSize, [&](TDataProviderPtr datasetPart) {
        auto processedDataProvider = CreateModelCompatibleProcessedDataProvider(
            *datasetPart,
            metricDescriptions,
            model,
            GetMonopolisticFreeCpuRam(),
            rand,
            executor);
        processedDatasetParts->push_back(std::move(processedDataProvider));
    },
    executor);
}


int mode_eval_metrics(int argc, const char* argv[]) {
    NCB::TAnalyticalModeCommonParams params;
    TModeEvalMetricsParams plotParams;
    bool verbose = false;

    auto parser = NLastGetopt::TOpts();
    parser.AddHelpOption();
    params.BindParserOpts(parser);
    plotParams.BindParserOpts(parser);
    parser.AddLongOption("verbose")
        .SetFlag(&verbose)
        .NoArgument();

    bool saveStats = false;
    parser.AddLongOption("save-stats")
        .SetFlag(&saveStats)
        .NoArgument();

    bool calcOnParts = false;
    parser.AddLongOption("calc-on-parts")
        .SetFlag(&calcOnParts)
        .NoArgument();

    parser.SetFreeArgsNum(0);
    {
        NLastGetopt::TOptsParseResult parseResult(&parser, argc, argv);
        Y_UNUSED(parseResult);
    }
    TSetLoggingVerboseOrSilent inThisScope(verbose);

    params.DatasetReadingParams.ValidatePoolParams();

    TFullModel model = ReadModel(params.ModelFileName, params.ModelFormat);
    CB_ENSURE(
        model.GetUsedCatFeaturesCount() == 0 || params.DatasetReadingParams.ColumnarPoolFormatParams.CdFilePath.Inited(),
        "Model has categorical features. Specify column_description file with correct categorical features.");
    params.DatasetReadingParams.ClassLabels = model.GetModelClassLabels();

    if (plotParams.EndIteration == 0) {
        plotParams.EndIteration = model.GetTreeCount();
    }
    if (plotParams.TmpDir == "-") {
        plotParams.TmpDir = TTempDir().Name();
    }

    NPar::TLocalExecutor executor;
    executor.RunAdditionalThreads(params.ThreadCount - 1);

    TRestorableFastRng64 rand(0);

    auto metricDescriptions = CreateMetricDescriptions(plotParams.MetricsDescription);
    auto metrics = CreateMetrics(metricDescriptions, model.GetDimensionsCount());

    TMetricsPlotCalcer plotCalcer = CreateMetricCalcer(
        model,
        plotParams.FirstIteration,
        plotParams.EndIteration,
        plotParams.Step,
        /*processedIterationsStep=*/50, // TODO(nikitxskv): Make auto estimation of this parameter based on the free RAM and pool size.
        plotParams.TmpDir,
        metrics,
        &executor
    );

    TVector<TProcessedDataProvider> datasetParts;
    if (plotCalcer.HasAdditiveMetric()) {
        ReadAndProceedPoolInBlocks(
            params.DatasetReadingParams,
            plotParams.ReadBlockSize,
            [&](TDataProviderPtr datasetPart) {
                auto processedDataProvider = CreateModelCompatibleProcessedDataProvider(
                    *datasetPart,
                    metricDescriptions,
                    model,
                    GetMonopolisticFreeCpuRam(),
                    &rand,
                    &executor);

                plotCalcer.ProceedDataSetForAdditiveMetrics(processedDataProvider);
                if (plotCalcer.HasNonAdditiveMetric() && !calcOnParts) {
                    datasetParts.push_back(std::move(processedDataProvider));
                }
            },
            &executor);
    }

    if (plotCalcer.HasNonAdditiveMetric() && calcOnParts) {
        while (!plotCalcer.AreAllIterationsProcessed()) {
            ReadAndProceedPoolInBlocks(
                params.DatasetReadingParams,
                plotParams.ReadBlockSize,
                [&](TDataProviderPtr datasetPart) {
                    auto processedDataProvider = CreateModelCompatibleProcessedDataProvider(
                        *datasetPart,
                        metricDescriptions,
                        model,
                        GetMonopolisticFreeCpuRam(),
                        &rand,
                        &executor);
                    plotCalcer.ProceedDataSetForNonAdditiveMetrics(processedDataProvider);
                },
                &executor);
            plotCalcer.FinishProceedDataSetForNonAdditiveMetrics();
        }
    }

    if (plotCalcer.HasNonAdditiveMetric() && !calcOnParts) {
        if (datasetParts.empty()) {
            ReadDatasetParts(params, plotParams.ReadBlockSize, metricDescriptions, model, &rand, &executor, &datasetParts);
        }
        plotCalcer.ComputeNonAdditiveMetrics(datasetParts);
    }
    plotCalcer.SaveResult(plotParams.ResultDirectory, params.OutputPath.Path, true /*saveMetrics*/, saveStats).ClearTempFiles();
    return 0;
}
