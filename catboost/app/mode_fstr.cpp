#include "cmd_line.h"
#include "output_fstr.h"

#include <catboost/libs/algo/calc_fstr.h>
#include <catboost/libs/data/load_data.h>
#include <catboost/libs/model/model.h>

#include <library/threading/local_executor/local_executor.h>
#include <library/getopt/small/last_getopt.h>

#include <util/system/fs.h>

int mode_fstr(int argc, const char* argv[]) {
    TAnalyticalModeCommonParams params;

    auto parser = NLastGetopt::TOpts();
    parser.AddHelpOption();
    params.BindParserOpts(parser);
    parser.AddLongOption("fstr-type", "Should be one of: FeatureImportance, InternalFeatureImportance, Interaction, InternalInteraction")
        .RequiredArgument("fstr-type")
        .Handler1T<TString>([&params](const TString& fstrType) {
            CB_ENSURE(TryFromString<EFstrType>(fstrType, params.FstrType), fstrType + " fstr type is not supported");
        });
    parser.SetFreeArgsNum(0);
    NLastGetopt::TOptsParseResult parserResult{&parser, argc, argv};

    CB_ENSURE(NFs::Exists(params.ModelFileName));
    TFullModel model = ReadModel(params.ModelFileName);
    CB_ENSURE(model.CtrCalcerData.LearnCtrs.empty() || !params.CdFile.empty(), "specify column_description file for fstr mode");

    TPool pool;
    ReadPool(params.CdFile, params.InputPath, params.ThreadCount, false, &pool);

    switch (params.FstrType) {
        case EFstrType::FeatureImportance:
            CalcAndOutputFstr(model, pool, &params.OutputPath, nullptr, params.ThreadCount);
            break;
        case EFstrType::InternalFeatureImportance:
            CalcAndOutputFstr(model, pool, nullptr, &params.OutputPath, params.ThreadCount);
            break;
        case EFstrType::Interaction:
            CalcAndOutputInteraction(model, pool, &params.OutputPath, nullptr);
            break;
        case EFstrType::InternalInteraction:
            CalcAndOutputInteraction(model, pool, nullptr, &params.OutputPath);
            break;
        case EFstrType::Doc:
            CalcAndOutputDocFstr(model, pool, params.OutputPath, params.ThreadCount);
            break;
        default:
            Y_ASSERT(false);
    }

    return 0;
}
