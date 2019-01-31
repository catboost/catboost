#include "mode_fstr_helpers.h"

#include <catboost/libs/data_new/load_data.h>
#include <catboost/libs/fstr/output_fstr.h>
#include <catboost/libs/fstr/shap_values.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/model/model.h>

#include <util/generic/ptr.h>
#include <util/generic/serialized_enum.h>
#include <util/string/cast.h>
#include <util/system/yassert.h>


namespace {
    class TLazyPoolLoader {
    public:
        TLazyPoolLoader(const NCB::TAnalyticalModeCommonParams& params,
                        const TFullModel& model,
                        TAtomicSharedPtr<NPar::TLocalExecutor> localExecutor)
            : Params(params)
            , Model(model)
            , LocalExecutor(std::move(localExecutor))
        {}

        const NCB::TDataProviderPtr operator()() {
            if (!Dataset) {
                /* TODO(akhropov): there's a possibility of pool format with cat features w/o cd file in the future,
                    so these checks might become wrong and cat features spec in pool should be checked instead
                */
                CB_ENSURE(Model.GetUsedCatFeaturesCount() == 0 || Params.DsvPoolFormatParams.CdFilePath.Inited(),
                          "Model has categorical features. Specify column_description file with correct categorical features.");
                if (Model.HasCategoricalFeatures()) {
                    CB_ENSURE(Params.DsvPoolFormatParams.CdFilePath.Inited(),
                              "Model has categorical features. Specify column_description file with correct categorical features.");
                }

                TSetLoggingVerboseOrSilent inThisScope(false);

                Dataset = NCB::ReadDataset(Params.InputPath,
                                           Params.PairsFilePath,
                                           /*groupWeightsFilePath=*/NCB::TPathWithScheme(),
                                           Params.DsvPoolFormatParams,
                                           /*ignoredFeatures*/ {},
                                           NCB::EObjectsOrder::Undefined,
                                           LocalExecutor.Get());
            }
            return Dataset;
        }
    private:
        const NCB::TAnalyticalModeCommonParams& Params;
        const TFullModel& Model;
        TAtomicSharedPtr<NPar::TLocalExecutor> LocalExecutor;

        NCB::TDataProviderPtr Dataset;
    };
}

void NCB::PrepareFstrModeParamsParser(
    NCB::TAnalyticalModeCommonParams* paramsPtr,
    NLastGetopt::TOpts* parserPtr) {

    auto& params = *paramsPtr;
    auto& parser = *parserPtr;

    parser.AddHelpOption();
    params.BindParserOpts(parser);
    parser.FindLongOption("output-path")
        ->DefaultValue("feature_strength.tsv");
    const auto customFstrTypeDescription = TString::Join("Should be one of: ", GetEnumAllNames<EFstrType >());
    parser.AddLongOption("fstr-type", customFstrTypeDescription)
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
}

void NCB::ModeFstrSingleHost(const NCB::TAnalyticalModeCommonParams& params) {

    TFullModel model = ReadModel(params.ModelFileName, params.ModelFormat);
    if (model.HasCategoricalFeatures()) {
        CB_ENSURE(model.HasValidCtrProvider(),
                  "Model has invalid ctr provider, possibly you are using core model without or with incomplete ctr data");
    }
    // TODO(noxoomo): have ignoredFeatures and const features saved in the model file

    auto localExecutor = MakeAtomicShared<NPar::TLocalExecutor>();
    localExecutor->RunAdditionalThreads(params.ThreadCount - 1);

    TLazyPoolLoader poolLoader(params, model, localExecutor);

    switch (params.FstrType) {
        case EFstrType::PredictionValuesChange:
            CalcAndOutputFstr(model,
                              model.ObliviousTrees.LeafWeights.empty() ? poolLoader() : nullptr,
                              localExecutor.Get(),
                              &params.OutputPath.Path,
                              nullptr,
                              params.FstrType);
            break;
        case EFstrType::LossFunctionChange:
            CalcAndOutputFstr(model,
                              poolLoader(),
                              localExecutor.Get(),
                              &params.OutputPath.Path,
                              nullptr,
                              params.FstrType);
            break;
        case EFstrType::InternalFeatureImportance:
            CalcAndOutputFstr(model,
                              model.ObliviousTrees.LeafWeights.empty() ? poolLoader() : nullptr,
                              localExecutor.Get(),
                              nullptr,
                              &params.OutputPath.Path,
                              params.FstrType);
            break;
        case EFstrType::Interaction:
            CalcAndOutputInteraction(model, &params.OutputPath.Path, nullptr);
            break;
        case EFstrType::InternalInteraction:
            CalcAndOutputInteraction(model, nullptr, &params.OutputPath.Path);
            break;
        case EFstrType::ShapValues:
            CalcAndOutputShapValues(model, *poolLoader(), params.OutputPath.Path, params.Verbose, localExecutor.Get());
            break;
        default:
            Y_ASSERT(false);
    }

}

