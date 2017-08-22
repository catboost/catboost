#include "cmd_line.h"

#include <catboost/libs/data/load_data.h>
#include <catboost/libs/algo/apply.h>
#include <catboost/libs/algo/eval_helpers.h>
#include <catboost/libs/model/model.h>

#include <library/getopt/small/last_getopt.h>
#include <library/threading/local_executor/local_executor.h>

#include <util/stream/file.h>
#include <util/system/fs.h>
#include <util/string/iterator.h>

int mode_calc(int argc, const char* argv[]) {
    TAnalyticalModeCommonParams params;
    int iterationsLimit = 0;

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
    parser.AddLongOption("class-names", "names for classes.")
        .RequiredArgument("comma separated list of names")
        .Handler1T<TString>([&params](const TString& namesLine) {
            for (const auto& t : StringSplitter(namesLine).Split(',')) {
                params.ClassNames.push_back(FromString<TString>(t.Token()));
            }
        });
    parser.SetFreeArgsNum(0);
    NLastGetopt::TOptsParseResult parserResult{&parser, argc, argv};

    CB_ENSURE(NFs::Exists(params.ModelFileName));
    TFullModel model = ReadModel(params.ModelFileName);
    CB_ENSURE(model.CtrCalcerData.LearnCtrs.empty() || !params.CdFile.empty(), "specify column_description file for calc mode");

    TPool pool;
    ReadPool(params.CdFile, params.InputPath, params.ThreadCount, false, '\t', false, params.ClassNames, &pool);

    yvector<yvector<double>> approx = ApplyModelMulti(model, pool, true, params.PredictionType, 0, iterationsLimit, params.ThreadCount);

    OutputTestEval(approx, params.OutputPath, pool.Docs, false);
    return 0;
}
