#include "modes.h"

#include <catboost/libs/data/loader.h>
#include <catboost/libs/data/data_provider_builders.h>
#include <catboost/libs/data/proceed_pool_in_blocks.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/model/model_export/model_exporter.h>
#include <catboost/libs/model/enums.h>
#include <catboost/private/libs/algo/apply.h>
#include <catboost/private/libs/app_helpers/mode_normalize_model_helpers.h>
#include <catboost/private/libs/options/analytical_mode_params.h>
#include <catboost/private/libs/options/dataset_reading_params.h>
#include <catboost/private/libs/data_util/path_with_scheme.h>

#include <library/getopt/small/last_getopt.h>
#include <library/json/json_value.h>
#include <library/threading/local_executor/local_executor.h>

#include <util/generic/serialized_enum.h>
#include <util/system/mutex.h>

using namespace NCB;

namespace {

    struct TModeParams {
        TString ModelFileName;
        EModelType ModelType = EModelType::CatboostBinary;
        TString OutputModelFileName;
        TMaybe<EModelType> OutputModelType;
        NCatboostOptions::TColumnarPoolFormatParams ColumnarPoolFormatParams;
        TMaybe<double> Scale;
        TMaybe<double> Bias;
        ELoggingLevel LoggingLevel;
        int ThreadCount;
        TVector<TPathWithScheme> PoolPaths;
        bool PrintScaleAndBias = false;

        TModeParams(int argc, const char* argv[]) {
            auto parser = NLastGetopt::TOpts();
            parser.AddHelpOption();
            parser.SetFreeArgsNum(0);
            BindModelFileParams(&parser, &ModelFileName, &ModelType);
            BindColumnarPoolFormatParams(&parser, &ColumnarPoolFormatParams);
            parser.AddLongOption("set-scale").RequiredArgument("SCALE")
                .Handler1T<double>([=](auto scale){ Scale = scale; })
                .Help("Scale")
                ;
            parser.AddLongOption("set-bias").RequiredArgument("BIAS")
                .Handler1T<double>([=](auto bias){ Bias = bias; })
                .Help("Bias")
                ;
            parser.AddLongOption("print-scale-and-bias").NoArgument()
                .StoreTrue(&PrintScaleAndBias)
                .Help("Print input and resulting scale and bias")
                ;
            parser.AddLongOption("logging-level").RequiredArgument("LEVEL")
                .Handler1T<TStringBuf>([=](auto level){ LoggingLevel = FromString<ELoggingLevel>(level); })
                .Help("Logging level, one of " + GetEnumAllNames<ELoggingLevel>())
                .DefaultValue(ELoggingLevel::Info)
                ;
            parser.AddLongOption('T', "thread-count").RequiredArgument("N")
                .StoreResult(&ThreadCount)
                .Help("Worker thread count")
                .DefaultValue(NSystemInfo::CachedNumberOfCpus())
                ;
            parser.AddLongOption('i', "input-path").RequiredArgument("PATH...")
                .Handler1T<TStringBuf>([=](auto path){ PoolPaths.push_back(TPathWithScheme{path, "dsv"}); })
                .Help("Pool path (repeat the option for multiple pools)")
                ;
            parser.AddLongOption("output-model").RequiredArgument("PATH")
                .StoreResult(&OutputModelFileName)
                .Help("Output model path")
                ;
            parser.AddLongOption("output-model-format").RequiredArgument("FORMAT")
                .Handler1T<TStringBuf>([=](auto format){ OutputModelType = FromString<EModelType>(format); })
                .Help("Output model format, one of " + GetEnumAllNames<EModelType>())
                ;
            NLastGetopt::TOptsParseResult parseResult{&parser, argc, argv};
        }
    };

    class TOpenSourceModeNormalizeModelImplementation : public IModeNormalizeModelImplementation {

        int mode_normalize_model(int argc, const char** argv) const override {

            TModeParams modeParams(argc, argv);
            TSetLogging inThisScope(modeParams.LoggingLevel);

            TFullModel model = ReadModel(modeParams.ModelFileName, modeParams.ModelType);
            CB_ENSURE(model.GetTreeCount() > 0, "Cannot normalize empty model");
            CB_ENSURE(model.GetDimensionsCount() == 1, "No sense in normalizing a multiclass/multiregression model");
            TScaleAndBias inputScaleAndBias = model.GetScaleAndBias();
            if (modeParams.PrintScaleAndBias) {
                Cout << "Input model"
                    << " scale " << inputScaleAndBias.Scale
                    << " bias " << inputScaleAndBias.Bias
                    << Endl;
            }

            if (modeParams.PoolPaths) {
                CB_ENSURE(!modeParams.Scale.Defined() && !modeParams.Bias.Defined(), "Conflicting options: -i and --set-scale/bias");
                model.SetScaleAndBias({1.0, 0.0});
                auto approx = CalcMinMaxOnAllPools(model, modeParams);
                CB_ENSURE(approx.Min < approx.Max, "Model gives same result on all docs");
                double scale = 1.0 / (approx.Max - approx.Min);
                double bias = - scale * approx.Min;
                model.SetScaleAndBias({scale, bias});
            } else {
                double scale = modeParams.Scale.GetOrElse(model.GetScaleAndBias().Scale);
                double bias = modeParams.Bias.GetOrElse(model.GetScaleAndBias().Bias);
                model.SetScaleAndBias({scale, bias});
            }

            if (inputScaleAndBias != model.GetScaleAndBias() || modeParams.OutputModelFileName) {
                if (modeParams.PrintScaleAndBias) {
                    Cout << "Output model"
                        << " scale " << model.GetScaleAndBias().Scale
                        << " bias " << model.GetScaleAndBias().Bias
                        << Endl;
                }
                const TString& outputModelFileName = modeParams.OutputModelFileName ? modeParams.OutputModelFileName : modeParams.ModelFileName;
                const EModelType outputModelType = modeParams.OutputModelType.GetOrElse(modeParams.ModelType);
                ExportModel(model, outputModelFileName, outputModelType);
            }
            return 0;
        }

        TMinMax<double> CalcMinMaxOnAllPools(
            const TFullModel& model,
            const TModeParams& modeParams
        ) const {
            NPar::TLocalExecutor localExecutor;
            localExecutor.RunAdditionalThreads(modeParams.ThreadCount - 1);
            TMinMax<double> result{+DBL_MAX, -DBL_MAX};
            for (auto poolPath : modeParams.PoolPaths) {
                auto approx = CalcMinMaxOnOnePool(model, poolPath, modeParams, &localExecutor);
                CATBOOST_DEBUG_LOG << "Pool " << poolPath.Path << ", approx" << " min " << approx.Min << " max " << approx.Max << Endl;
                result.Min = Min(result.Min, approx.Min);
                result.Max = Max(result.Max, approx.Max);
            }
            return result;
        }

        TMinMax<double> CalcMinMaxOnOnePool(
            const TFullModel& model,
            const TPathWithScheme& poolPath,
            const TModeParams& modeParams,
            NPar::TLocalExecutor* localExecutor
        ) const {
            TMinMax<double> result{+DBL_MAX, -DBL_MAX};
            TMutex result_guard;
            ReadAndProceedPoolInBlocks(
                NCatboostOptions::TDatasetReadingParams{
                    modeParams.ColumnarPoolFormatParams,
                    poolPath,
                    TVector<NJson::TJsonValue>(),  // ClassLabels
                    TPathWithScheme(),  // PairsFilePath
                    TPathWithScheme(),  // FeatureNamesPath
                    TVector<ui32>(),  // IgnoredFeatures
                },
                10000,  // blockSize
                [&](const TDataProviderPtr datasetPart) {
                    auto approx = ApplyModelForMinMax(
                        model,
                        *datasetPart->ObjectsData,
                        0, // treeBegin
                        0, // treeEnd
                        nullptr // localExecutor
                    );
                    GuardedUpdateMinMax(approx, &result, result_guard);
                },
                localExecutor
            );
            return result;
        }
    };
}

TModeNormalizeModelImplementationFactory::TRegistrator<TOpenSourceModeNormalizeModelImplementation> YandexSpecificModeNormalizeModelImplementationRegistrator(EImplementationType::OpenSource);
