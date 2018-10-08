#include "modes.h"
#include "cmd_line.h"
#include "bind_options.h"
#include "proceed_pool_in_blocks.h"

#include <catboost/libs/data/load_data.h>
#include <catboost/libs/algo/apply.h>
#include <catboost/libs/eval_result/eval_result.h>
#include <catboost/libs/labels/external_label_helper.h>
#include <catboost/libs/model/model.h>

#include <library/getopt/small/last_getopt.h>
#include <library/threading/local_executor/local_executor.h>

#include <util/stream/file.h>
#include <util/system/fs.h>
#include <util/string/iterator.h>

#include <algorithm>

using NCB::TEvalResult;
using NCB::ValidateColumnOutput;

static TEvalResult Apply(
    const TFullModel& model,
    const TPool& pool,
    size_t begin, size_t end,
    size_t evalPeriod,
    NPar::TLocalExecutor* executor)
{
    TEvalResult resultApprox;
    TVector<TVector<TVector<double>>>& rawValues = resultApprox.GetRawValuesRef();
    rawValues.resize(1);
    if (pool.Docs.Baseline.ysize() > 0) {
        rawValues[0].assign(pool.Docs.Baseline.begin(), pool.Docs.Baseline.end());
    } else {
        rawValues[0].resize(model.ObliviousTrees.ApproxDimension, TVector<double>(pool.Docs.GetDocCount(), 0.0));
    }
    TModelCalcerOnPool modelCalcerOnPool(model, pool, *executor);
    TVector<double> flatApprox;
    TVector<TVector<double>> approx;
    for (; begin < end; begin += evalPeriod) {
        modelCalcerOnPool.ApplyModelMulti(EPredictionType::InternalRawFormulaVal,
                                          begin,
                                          Min(begin + evalPeriod, end),
                                          &flatApprox,
                                          &approx);

        for (size_t i = 0; i < approx.size(); ++i) {
            for (size_t j = 0; j < approx[0].size(); ++j) {
                rawValues.back()[i][j] += approx[i][j];
            }
        }
        if (begin + evalPeriod < end) {
            rawValues.push_back(rawValues.back());
        }
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
    parser.AddLongOption("output-columns")
        .RequiredArgument("Comma separated list of column indexes")
        .Handler1T<TString>([&](const TString& outputColumns) {
            params.OutputColumnsIds.clear();
            for (const auto& typeName : StringSplitter(outputColumns).Split(',')) {
                params.OutputColumnsIds.push_back(FromString<TString>(typeName.Token()));
            }
        });
    parser.AddLongOption("prediction-type")
        .RequiredArgument(
            "Comma separated list of prediction types. Every prediction type should be one of: Probability, Class, RawFormulaVal")
        .Handler1T<TString>([&](const TString& predictionTypes) {
            params.PredictionTypes.clear();
            params.OutputColumnsIds = {"DocId"};
            for (const auto& typeName : StringSplitter(predictionTypes).Split(',')) {
                params.PredictionTypes.push_back(FromString<EPredictionType>(typeName.Token()));
                params.OutputColumnsIds.push_back(FromString<TString>(typeName.Token()));
            }
        });
    parser.AddLongOption("eval-period", "predictions are evaluated every <eval-period> trees")
        .StoreResult(&evalPeriod);
    parser.SetFreeArgsNum(0);
    NLastGetopt::TOptsParseResult parserResult{&parser, argc, argv};

    TFullModel model = ReadModel(params.ModelFileName, params.ModelFormat);
    if (model.HasCategoricalFeatures()) {
        CB_ENSURE(params.DsvPoolFormatParams.CdFilePath.Inited(),
                  "Model has categorical features. Specify column_description file with correct categorical features.");
        CB_ENSURE(model.HasValidCtrProvider(),
                  "Model has invalid ctr provider, possibly you are using core model without or with incomplete ctr data");
    }
    params.ClassNames = ReadClassNames(model.ModelInfo.at("params"));

    if (iterationsLimit == 0) {
        iterationsLimit = model.GetTreeCount();
    }
    iterationsLimit = Min(iterationsLimit, model.GetTreeCount());

    if (evalPeriod == 0) {
        evalPeriod = iterationsLimit;
    } else {
        evalPeriod = Min(evalPeriod, iterationsLimit);
    }

    const int blockSize = Max<int>(32, static_cast<int>(10000. / (static_cast<double>(iterationsLimit) / evalPeriod) / model.ObliviousTrees.ApproxDimension));
    TOFStream outputStream(params.OutputPath.Path);
    NPar::TLocalExecutor executor;
    executor.RunAdditionalThreads(params.ThreadCount - 1);

    SetVerboseLogingMode();
    bool IsFirstBlock = true;
    ui64 docIdOffset = 0;
    auto poolColumnsPrinter = CreatePoolColumnPrinter(params.InputPath, params.DsvPoolFormatParams.Format);
    ReadAndProceedPoolInBlocks(params, blockSize, [&](const TPool& poolPart) {
        if (IsFirstBlock) {
            ValidateColumnOutput(params.OutputColumnsIds, poolPart, true);
        }
        auto approx = Apply(model, poolPart, 0, iterationsLimit, evalPeriod, &executor);
        auto visibleLabelsHelper = BuildLabelsHelper<TExternalLabelsHelper>(model);

        poolColumnsPrinter->UpdateColumnTypeInfo(poolPart.MetaInfo.ColumnsInfo);

        SetSilentLogingMode();
        OutputEvalResultToFile(
            approx,
            &executor,
            params.OutputColumnsIds,
            visibleLabelsHelper,
            poolPart,
            true,
            &outputStream,
            // TODO: src file columns output is incompatible with block processing
            poolColumnsPrinter,
            /*testFileWhichOf*/ {0, 0},
            IsFirstBlock,
            docIdOffset,
            std::make_pair(evalPeriod, iterationsLimit)
        );
        docIdOffset += blockSize;
        IsFirstBlock = false;
    }, &executor);

    return 0;
}
