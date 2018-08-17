#include "modes.h"
#include "cmd_line.h"

#include <catboost/libs/fstr/shap_values.h>
#include <catboost/libs/fstr/calc_fstr.h>
#include <catboost/libs/fstr/output_fstr.h>
#include <catboost/libs/data/load_data.h>
#include <catboost/libs/model/model.h>

#include <library/threading/local_executor/local_executor.h>
#include <library/getopt/small/last_getopt.h>

#include <util/generic/ptr.h>
#include <util/system/fs.h>
#include <util/string/iterator.h>


class TLazyPoolLoader {
public:
    TLazyPoolLoader(const TAnalyticalModeCommonParams& params,
                    const TFullModel& model)
        : Params(params)
        , Model(model)
    {}

    const TPool& operator()() {
        if (!Pool) {
            /* TODO(akhropov): there's a possibility of pool format with cat features w/o cd file in the future,
                so these checks might become wrong and cat features spec in pool should be checked instead
            */
            CB_ENSURE(Model.ObliviousTrees.CatFeatures.empty() || Params.DsvPoolFormatParams.CdFilePath.Inited(),
                      "Model has categorical features. Specify column_description file with correct categorical features.");
            if (Model.HasCategoricalFeatures()) {
                CB_ENSURE(Params.DsvPoolFormatParams.CdFilePath.Inited(),
                          "Model has categorical features. Specify column_description file with correct categorical features.");
            }

            NCB::TTargetConverter targetConverter = NCB::MakeTargetConverter(Params.ClassNames);

            Pool.Reset(new TPool);
            NCB::ReadPool(Params.InputPath,
                          Params.PairsFilePath,
                          /*groupWeightsFilePath=*/NCB::TPathWithScheme(),
                          Params.DsvPoolFormatParams,
                          /*ignoredFeatures*/ {},
                          Params.ThreadCount,
                          false,
                          &targetConverter,
                          Pool.Get());
        }
        return *Pool;
    }
private:
    const TAnalyticalModeCommonParams& Params;
    const TFullModel& Model;

    THolder<TPool> Pool;
};


int mode_fstr(int argc, const char* argv[]) {
    TAnalyticalModeCommonParams params;

    auto parser = NLastGetopt::TOpts();
    parser.AddHelpOption();
    params.BindParserOpts(parser);
    parser.FindLongOption("output-path")
        ->DefaultValue("feature_strength.tsv");
    parser.AddLongOption("fstr-type", "Should be one of: FeatureImportance, InternalFeatureImportance, Interaction, InternalInteraction, ShapValues")
        .RequiredArgument("fstr-type")
        .Handler1T<TString>([&params](const TString& fstrType) {
            CB_ENSURE(TryFromString<EFstrType>(fstrType, params.FstrType), fstrType + " fstr type is not supported");
        });
    parser.AddLongOption("verbose", "Log writing period")
        .DefaultValue("0")
        .Handler1T<TString>([&params](const TString& verbose) {
            CB_ENSURE(TryFromString<int>(verbose, params.Verbose), "verbose should be integer");
            CB_ENSURE(params.Verbose >= 0, "verbose should be non-negative");
        });
    parser.SetFreeArgsNum(0);
    NLastGetopt::TOptsParseResult parserResult{&parser, argc, argv};

    CB_ENSURE(NFs::Exists(params.ModelFileName), "Model file doesn't exist: " << params.ModelFileName);
    TFullModel model = ReadModel(params.ModelFileName);
    if (model.HasCategoricalFeatures()) {
        CB_ENSURE(model.HasValidCtrProvider(),
                  "Model has invalid ctr provider, possibly you are using core model without or with incomplete ctr data");
    }
    // TODO(noxoomo): have ignoredFeatures and const features saved in the model file

    TLazyPoolLoader poolLoader(params, model);

    switch (params.FstrType) {
        case EFstrType::FeatureImportance:
            CalcAndOutputFstr(model,
                              model.ObliviousTrees.LeafWeights.empty() ? &(poolLoader()) : nullptr,
                              &params.OutputPath,
                              nullptr);
            break;
        case EFstrType::InternalFeatureImportance:
            CalcAndOutputFstr(model,
                              model.ObliviousTrees.LeafWeights.empty() ? &(poolLoader()) : nullptr,
                              nullptr,
                              &params.OutputPath);
            break;
        case EFstrType::Interaction:
            CalcAndOutputInteraction(model, &params.OutputPath, nullptr);
            break;
        case EFstrType::InternalInteraction:
            CalcAndOutputInteraction(model, nullptr, &params.OutputPath);
            break;
        case EFstrType::ShapValues:
            CalcAndOutputShapValues(model, poolLoader(), params.OutputPath, params.ThreadCount, params.Verbose);
            break;
        default:
            Y_ASSERT(false);
    }

    return 0;
}
