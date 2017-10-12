#include "cmd_line.h"
#include "proceed_pool_in_blocks.h"

#include <catboost/libs/data/load_data.h>
#include <catboost/libs/algo/apply.h>
#include <util/system/fs.h>
#include <util/string/iterator.h>
#include <catboost/libs/algo/plot.h>

struct TModePlotParams {
    ui32 Step = 0;
    ui32 FirstIteration = 0;
    ui32 LastIteration = 0;
    int ReadBlockSize = 32768;
    TString MetricsDescription;
    TString TmpDir;

    void BindParserOpts(NLastGetopt::TOpts& parser) {
        parser.AddLongOption("first-iteration", "First iteration to plot.")
                .RequiredArgument("INT")
                .StoreResult(&FirstIteration);
        parser.AddLongOption("last-iteration", "Last iteration to plot.")
                .RequiredArgument("INT")
                .StoreResult(&LastIteration);
        parser.AddLongOption("step", "Plot every step trees.")
                .RequiredArgument("INT")
                .StoreResult(&Step);
        parser.AddLongOption("eval-metric", "coma-separated eval metrics")
                .RequiredArgument("String")
                .StoreResult(&MetricsDescription);
        parser.AddLongOption("block-size", "Compute block size")
                .RequiredArgument("INT")
                .DefaultValue("32768")
                .StoreResult(&ReadBlockSize);
        parser.AddLongOption("tmp-dir", "Dir to store approx for non-additive metrics")
                .RequiredArgument("INT")
                .DefaultValue("tmp")
                .StoreResult(&TmpDir);
    }
};

int mode_plot(int argc, const char* argv[]) {
    TAnalyticalModeCommonParams params;
    TModePlotParams plotParams;
    bool verbose = false;

    auto parser = NLastGetopt::TOpts();
    parser.AddHelpOption();
    params.BindParserOpts(parser);
    plotParams.BindParserOpts(parser);
    parser.AddLongOption("verbose", "Dir to store approx for non-additive metrics")
        .SetFlag(&verbose)
        .NoArgument();
    parser.SetFreeArgsNum(0);
    {
        NLastGetopt::TOptsParseResult parseResult(&parser, argc, argv);
        Y_UNUSED(parseResult);
    }
    if (verbose) {
        SetVerboseLogingMode();
    } else {
        SetSilentLogingMode();
    }

    CB_ENSURE(NFs::Exists(params.ModelFileName), "Model file doesn't exist " << params.ModelFileName);
    TFullModel model = ReadModel(params.ModelFileName);
    CB_ENSURE(model.CtrCalcerData.LearnCtrs.empty() || !params.CdFile.empty(), "Model has categorical features. Specify column_description file with correct categorical features.");
    if (plotParams.LastIteration == 0) {
        plotParams.LastIteration = model.TreeStruct.size();
    }
    if (plotParams.Step == 0) {
        plotParams.Step = (plotParams.LastIteration - plotParams.FirstIteration) > 100 ? 10 : 1;
    }

    yvector<THolder<IMetric>> metrics;

    for (const auto& metricDescription : StringSplitter(plotParams.MetricsDescription).Split(',')) {
        TString metricStr = TString(metricDescription.Token());
        auto metricsBatch = CreateMetricFromDescription(metricStr, model.ApproxDimension);
        for (ui32 i = 0; i < metricsBatch.size(); ++i) {
            metrics.push_back(std::move(metricsBatch[i]));
        }
    }

    NPar::TLocalExecutor executor;
    executor.RunAdditionalThreads(params.ThreadCount - 1);
    TMetricsPlotCalcer plotCalcer(model, executor, plotParams.TmpDir);
    plotCalcer
        .SetFirstIteration(plotParams.FirstIteration)
        .SetLastIteration(plotParams.LastIteration)
        .SetCustomStep(plotParams.Step);

    for (auto& metric : metrics) {
        plotCalcer.AddMetric(*metric);
    }

    ReadAndProceedPoolInBlocks(params, plotParams.ReadBlockSize, [&](const TPool& poolPart) {
        plotCalcer.ProceedDataSet(poolPart);
    });

    TOFStream outputStream(params.OutputPath);
    plotCalcer.SaveResult(&outputStream)
            .ClearTempFiles();

    return 0;
}
