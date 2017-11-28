#include "cmd_line.h"
#include "proceed_pool_in_blocks.h"

#include <catboost/libs/data/load_data.h>
#include <catboost/libs/algo/apply.h>
#include <catboost/libs/helpers/eval_helpers.h>
#include <catboost/libs/model/model.h>

#include <library/getopt/small/last_getopt.h>
#include <library/threading/local_executor/local_executor.h>

#include <util/stream/file.h>
#include <util/system/fs.h>
#include <util/string/iterator.h>

#include <algorithm>

static TEvalResult Apply(
    const TFullModel& model,
    const TPool& pool,
    size_t begin, size_t end,
    size_t evalPeriod,
    NPar::TLocalExecutor* executor)
{
    TEvalResult resultApprox;
    TVector<TVector<double>>& rawValues = resultApprox.GetRawValuesRef();
    if (pool.Docs.Baseline.ysize() > 0) {
        rawValues.assign(pool.Docs.Baseline.begin(), pool.Docs.Baseline.end());
    }
    for (; begin < end; begin += evalPeriod) {
        TVector<TVector<double>> approx = ApplyModelMulti(model, pool, EPredictionType::RawFormulaVal,
                                                          begin, Min(begin + evalPeriod, end), *executor);
        if (rawValues.empty()) {
            rawValues.swap(approx);
        } else {
            for (size_t i = 0; i < approx.size(); ++i) {
                for (size_t j = 0; j < approx[0].size(); ++j) {
                    rawValues[i][j] += approx[i][j];
                }
            }
        }
        resultApprox.PostProcess(executor, std::make_pair((int)begin, (int)Min(begin + evalPeriod, end)));
    }
    return resultApprox;
}

int mode_calc(int argc, const char* argv[]) {
    TAnalyticalModeCommonParams params;
    size_t iterationsLimit = 0;
    size_t evalPeriod = 0;

    auto parser = NLastGetopt::TOpts();
    parser.AddHelpOption();
    params.BindParserOpts(parser);
    parser.AddLongOption("tree-count-limit", "limit count of used trees")
        .StoreResult(&iterationsLimit);
    parser.AddLongOption("prediction-type")
        .RequiredArgument("Comma separated list of prediction types. Every prediction type should be one of: Probability, Class, RawFormulaVal")
        .Handler1T<TString>([&](const TString& predictionTypes) {
            params.PredictionTypes.clear();
            for (const auto&  typeName : StringSplitter(predictionTypes).Split(',')) {
                params.PredictionTypes.push_back(FromString<EPredictionType>(typeName.Token()));
            }
        });
    parser.AddLongOption("eval-period", "predictions are evaluated every <eval-period> trees")
        .StoreResult(&evalPeriod);
    parser.SetFreeArgsNum(0);
    NLastGetopt::TOptsParseResult parserResult{&parser, argc, argv};

    CB_ENSURE(NFs::Exists(params.ModelFileName), "Model file doesn't exist " << params.ModelFileName);
    TFullModel model = ReadModel(params.ModelFileName);
    if (model.HasCategoricalFeatures()) {
        CB_ENSURE(!params.CdFile.empty(),
                  "Model has categorical features. Specify column_description file with correct categorical features.");
        CB_ENSURE(model.HasValidCtrProvider(),
                  "Model has invalid ctr provider, possibly you are using core model without or with incomplete ctr data");
    }

    if (iterationsLimit == 0) {
        iterationsLimit = model.GetTreeCount();
    }
    iterationsLimit = Min(iterationsLimit, model.GetTreeCount());

    if (evalPeriod == 0) {
        evalPeriod = model.GetTreeCount();
    } else {
        evalPeriod = Min(evalPeriod, model.GetTreeCount());
    }

    const int blockSize = Max<int>(32, static_cast<int>(10000. / (static_cast<double>(iterationsLimit) / evalPeriod) / model.ObliviousTrees.ApproxDimension));
    TOFStream outputStream(params.OutputPath);
    NPar::TLocalExecutor executor;
    executor.RunAdditionalThreads(params.ThreadCount - 1);

    SetVerboseLogingMode();
    bool IsFirstBlock = true;
    ReadAndProceedPoolInBlocks(params, blockSize, [&](const TPool& poolPart) {
        TEvalResult approx = Apply(model, poolPart, 0, iterationsLimit, evalPeriod, &executor);
        approx.SetPredictionTypes(params.PredictionTypes);
        SetSilentLogingMode();
        approx.PostProcess(&executor);
        approx.OutputToFile(poolPart.Docs.Id, &outputStream, IsFirstBlock);
        IsFirstBlock = false;
    });

    return 0;
}
