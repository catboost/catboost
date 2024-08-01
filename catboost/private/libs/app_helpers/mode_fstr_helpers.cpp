#include "mode_fstr_helpers.h"

#include <catboost/libs/data/load_data.h>
#include <catboost/libs/data/model_dataset_compatibility.h>
#include <catboost/libs/fstr/compare_documents.h>
#include <catboost/libs/fstr/output_fstr.h>
#include <catboost/libs/fstr/sage_values.h>
#include <catboost/libs/fstr/shap_values.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/libs/helpers/exception.h>

#include <library/cpp/getopt/small/last_getopt.h>

#include <util/folder/path.h>
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
                if (Model.HasCategoricalFeatures()) {
                    CB_ENSURE(
                        Params.DatasetReadingParams.ColumnarPoolFormatParams.CdFilePath.Inited(),
                        "Model has categorical features. Specify column_description file with correct categorical features.");
                }

                TSetLoggingVerboseOrSilent inThisScope(false);

                Dataset = NCB::ReadDataset(/*taskType*/Nothing(),
                                           Params.DatasetReadingParams.PoolPath,
                                           Params.DatasetReadingParams.PairsFilePath,
                                           Params.DatasetReadingParams.GraphFilePath,
                                           /*groupWeightsFilePath=*/NCB::TPathWithScheme(),
                                           /*baselineFilePath=*/ NCB::TPathWithScheme(),
                                           /*timestampsFilePath=*/ NCB::TPathWithScheme(),
                                           /*featureNamesPath=*/ NCB::TPathWithScheme(),
                                           /*poolMetaInfoPath=*/ NCB::TPathWithScheme(),
                                           Params.DatasetReadingParams.ColumnarPoolFormatParams,
                                           /*ignoredFeatures*/ {},
                                           NCB::EObjectsOrder::Undefined,
                                           NCB::TDatasetSubset::MakeColumns(),
                                           /*loadSampleIds*/ false,
                                           /*forceUnitAutoPairWeights*/ false,
                                           /*classLabels*/ Nothing(),
                                           LocalExecutor.Get());
                CheckModelAndDatasetCompatibility(Model, *Dataset->ObjectsData.Get());
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

    const auto customShapCalculationTypeDescription =
            TString::Join("Should be one of: ", GetEnumAllNames<ECalcTypeShapValues>());
    parser.AddLongOption("shap-calc-type")
        .DefaultValue("Regular")
        .Handler1T<TString>([&params](const TString& calcType) {
            CB_ENSURE(TryFromString<ECalcTypeShapValues>(calcType, params.ShapCalcType),
                    calcType + " shap calculation type is not supported");
        });

    parser.AddLongOption("verbose", "Log writing period")
        .DefaultValue("0")
        .Handler1T<TString>([&params](const TString& verbose) {
            CB_ENSURE(TryFromString<int>(verbose, params.Verbose), "verbose should be integer");
            CB_ENSURE(params.Verbose >= 0, "verbose should be non-negative");
        });
    parser.SetFreeArgsNum(0);
}

void NCB::ModeFstrSingleHostInner(
    const NCB::TAnalyticalModeCommonParams& params,
    const TFullModel& model) {

    auto fstrType = AdjustFeatureImportanceType(params.FstrType, model.GetLossFunctionName());
    if (fstrType != EFstrType::PredictionValuesChange) {
        CB_ENSURE_SCALE_IDENTITY(model.GetScaleAndBias(), "model fstr");
    }

    bool isInternalFstr = IsInternalFeatureImportanceType(params.FstrType);
    const TString* fstrPathPtr = isInternalFstr ? nullptr : &(params.OutputPath.Path);
    const TString* internalFstrPathPtr = !isInternalFstr ? nullptr : &(params.OutputPath.Path);

    auto localExecutor = MakeAtomicShared<NPar::TLocalExecutor>();
    localExecutor->RunAdditionalThreads(params.ThreadCount - 1);

    TLazyPoolLoader poolLoader(params, model, localExecutor);
    TFsPath inputPath(params.DatasetReadingParams.PoolPath.Path);

    switch (fstrType) {
        case EFstrType::PredictionValuesChange:
            CalcAndOutputFstr(model,
                              params.DatasetReadingParams.PoolPath.Inited() ? poolLoader() : nullptr,
                              localExecutor.Get(),
                              fstrPathPtr,
                              internalFstrPathPtr,
                              params.FstrType);
            break;
        case EFstrType::LossFunctionChange:
            CalcAndOutputFstr(model,
                              poolLoader(),
                              localExecutor.Get(),
                              fstrPathPtr,
                              internalFstrPathPtr,
                              params.FstrType);
            break;
        case EFstrType::Interaction:
            CalcAndOutputInteraction(model, fstrPathPtr, internalFstrPathPtr);
            break;
        case EFstrType::ShapValues:
            CalcAndOutputShapValues(model,
                                    *poolLoader(),
                                    params.OutputPath.Path,
                                    params.Verbose,
                                    EPreCalcShapValues::Auto,
                                    localExecutor.Get(),
                                    params.ShapCalcType);
            break;
        case EFstrType::SageValues:
            CalcAndOutputSageValues(model,
                                    *poolLoader(),
                                    params.Verbose,
                                    params.OutputPath.Path,
                                    localExecutor.Get());
            break;
        case EFstrType::PredictionDiff:
            CalcAndOutputPredictionDiff(
                model,
                *poolLoader(),
                params.OutputPath.Path,
                localExecutor.Get());
            break;
        default:
            Y_ASSERT(false);
    }
}

void NCB::ModeFstrSingleHost(const NCB::TAnalyticalModeCommonParams& params) {
    params.DatasetReadingParams.ValidatePoolParams();

    CB_ENSURE(params.ModelFileName.size() == 1, "Fstr calculation requires exactly one model");
    TFullModel model = ReadModel(params.ModelFileName[0], params.ModelFormat);
    if (model.HasCategoricalFeatures()) {
        CB_ENSURE(model.HasValidCtrProvider(),
                  "Model has invalid ctr provider, possibly you are using core model without or with incomplete ctr data");
    }
    // TODO: have ignoredFeatures and const features saved in the model file

    ModeFstrSingleHostInner(params, model);
}
