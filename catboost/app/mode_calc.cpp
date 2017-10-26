#include "cmd_line.h"
#include "proceed_pool_in_blocks.h"

#include <catboost/libs/data/load_data.h>
#include <catboost/libs/algo/apply.h>
#include <catboost/libs/algo/eval_helpers.h>
#include <catboost/libs/model/model.h>

#include <library/getopt/small/last_getopt.h>
#include <library/threading/local_executor/local_executor.h>

#include <util/stream/file.h>
#include <util/system/fs.h>
#include <util/string/iterator.h>

#include <algorithm>

// Returns approx[evalPeriodIdx][classIdx][objectIdx]
static yvector<yvector<yvector<double>>> Apply(const NCatBoost::TFormulaEvaluator& calcer,
                                               const TPool& pool,
                                               const EPredictionType predictionType, int begin, int end,
                                               int evalPeriod,
                                               NPar::TLocalExecutor* executor)
{
    yvector<yvector<double>> currentApprox; // [classIdx][objectIdx]
    yvector<yvector<yvector<double>>> resultApprox; // [evalPeriodIdx][classIdx][objectIdx]
    for (; begin < end; begin += evalPeriod) {
        yvector<yvector<double>> approx = ApplyModelMulti(calcer, pool, EPredictionType::RawFormulaVal,
                                                          begin, Min(begin + evalPeriod, end), *executor);
        if (currentApprox.empty()) {
            currentApprox.swap(approx);
        } else {
            for (size_t i = 0; i < approx.size(); ++i) {
                for (size_t j = 0; j < approx[0].size(); ++j) {
                    currentApprox[i][j] += approx[i][j];
                }
            }
        }
        resultApprox.push_back(PrepareEval(predictionType, currentApprox, executor));
    }
    return resultApprox;
}

int mode_calc(int argc, const char* argv[]) {
    TAnalyticalModeCommonParams params;
    int iterationsLimit = 0;
    int evalPeriod = 0;

    auto parser = NLastGetopt::TOpts();
    parser.AddHelpOption();
    params.BindParserOpts(parser);
    parser.AddLongOption("tree-count-limit", "limit count of used trees")
        .StoreResult(&iterationsLimit);
    parser.AddLongOption("prediction-type", "Should be one of: Probability, Class, RawFormulaVal")
        .RequiredArgument("prediction-type")
        .Handler1T<TString>([&params](const TString& predictionType) {
            params.PredictionType = FromString<EPredictionType>(predictionType);
        });
    parser.AddLongOption("eval-period", "predictions are evaluated every <eval-period> trees")
        .StoreResult(&evalPeriod);
    parser.SetFreeArgsNum(0);
    NLastGetopt::TOptsParseResult parserResult{&parser, argc, argv};

    CB_ENSURE(NFs::Exists(params.ModelFileName), "Model file doesn't exist " << params.ModelFileName);
    NCatBoost::TFormulaEvaluator calcer(ReadModel(params.ModelFileName));
    if (calcer.HasCategoricalFeatures()) {
        CB_ENSURE(!params.CdFile.empty(),
                  "Model has categorical features. Specify column_description file with correct categorical features.");
        CB_ENSURE(calcer.HasValidCtrProvider(),
                  "Model has invalid ctr provider, possibly you are using core model without or with incomplete ctr data");
    }

    if (iterationsLimit == 0) {
        iterationsLimit = calcer.GetTreeCount();
    }
    iterationsLimit = Min(iterationsLimit, calcer.GetTreeCount());

    if (evalPeriod == 0) {
        evalPeriod = calcer.GetTreeCount();
    } else {
        evalPeriod = Min(evalPeriod, calcer.GetTreeCount());
    }

    const int blockSize = Max<int>(32, static_cast<int>(10000. / (static_cast<double>(iterationsLimit) / evalPeriod) / calcer.GetModelClassCount()));
    TOFStream outputStream(params.OutputPath);
    NPar::TLocalExecutor executor;
    executor.RunAdditionalThreads(params.ThreadCount - 1);

    SetVerboseLogingMode();
    ReadAndProceedPoolInBlocks(params, blockSize, [&](const TPool& poolPart) {
        yvector<yvector<yvector<double>>> approx = Apply(calcer, poolPart, params.PredictionType,
                                                         0, iterationsLimit, evalPeriod, &executor);
        SetSilentLogingMode();
        OutputTestEval(approx, poolPart.Docs, false, &outputStream);
    });

    return 0;
}
