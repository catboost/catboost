#include "modes.h"
#include "cmd_line.h"
#include "output_fstr.h"

#include <catboost/libs/fstr/shap_values.h>
#include <catboost/libs/fstr/calc_fstr.h>
#include <catboost/libs/data/load_data.h>
#include <catboost/libs/model/model.h>

#include <library/threading/local_executor/local_executor.h>
#include <library/getopt/small/last_getopt.h>

#include <util/system/fs.h>
#include <util/string/iterator.h>

int mode_fstr(int argc, const char* argv[]) {
    TAnalyticalModeCommonParams params;

    auto parser = NLastGetopt::TOpts();
    parser.AddHelpOption();
    params.BindParserOpts(parser);
    parser.FindLongOption("output-path")
        ->DefaultValue("feature_strength.tsv");
    parser.AddLongOption("fstr-type", "Should be one of: FeatureImportance, InternalFeatureImportance, Interaction, InternalInteraction, Doc, ShapValues")
        .RequiredArgument("fstr-type")
        .Handler1T<TString>([&params](const TString& fstrType) {
            CB_ENSURE(TryFromString<EFstrType>(fstrType, params.FstrType), fstrType + " fstr type is not supported");
        });
    parser.SetFreeArgsNum(0);
    NLastGetopt::TOptsParseResult parserResult{&parser, argc, argv};

    CB_ENSURE(NFs::Exists(params.ModelFileName), "Model file doesn't exist: " << params.ModelFileName);
    TFullModel model = ReadModel(params.ModelFileName);
    CB_ENSURE(model.ObliviousTrees.CatFeatures.empty() || !params.CdFile.empty(), "Model has categorical features. Specify column_description file with correct categorical features.");
    if (model.HasCategoricalFeatures()) {
        CB_ENSURE(!params.CdFile.empty(),
                  "Model has categorical features. Specify column_description file with correct categorical features.");
        CB_ENSURE(model.HasValidCtrProvider(),
                  "Model has invalid ctr provider, possibly you are using core model without or with incomplete ctr data");
    }
    TPool pool;
    ReadPool(params.CdFile, params.InputPath, params.PairsFile, /*ignoredFeatures*/ {}, params.ThreadCount, false, params.Delimiter, params.HasHeader, params.ClassNames, &pool);
    // TODO(noxoomo): have ignoredFeatures and const features saved in the model file

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
        case EFstrType::ShapValues:
            CalcAndOutputShapValues(model, pool, params.OutputPath, params.ThreadCount);
            break;
        default:
            Y_ASSERT(false);
    }

    return 0;
}
