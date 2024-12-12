#include "bind_options.h"

#include <catboost/libs/column_description/column.h>
#include <catboost/libs/data/baseline.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/private/libs/options/analytical_mode_params.h>
#include <catboost/private/libs/options/dataset_reading_params.h>
#include <catboost/private/libs/options/catboost_options.h>
#include <catboost/private/libs/options/enums.h>
#include <catboost/private/libs/options/enum_helpers.h>
#include <catboost/private/libs/options/output_file_options.h>
#include <catboost/private/libs/options/plain_options_helper.h>

#include <library/cpp/getopt/small/last_getopt.h>
#include <library/cpp/grid_creator/binarization.h>
#include <library/cpp/json/json_reader.h>
#include <library/cpp/logger/log.h>
#include <library/cpp/text_processing/dictionary/options.h>

#include <util/generic/algorithm.h>
#include <util/generic/serialized_enum.h>
#include <util/generic/strbuf.h>
#include <util/generic/vector.h>
#include <util/string/cast.h>
#include <util/string/join.h>
#include <util/string/split.h>
#include <util/stream/file.h>
#include <util/system/fs.h>
#include <util/system/execpath.h>
#include <util/system/yassert.h>


using namespace NCB;

void InitOptions(
    const TString& optionsFile,
    NJson::TJsonValue* catBoostJsonOptions,
    NJson::TJsonValue* outputOptionsJson,
    NJson::TJsonValue* featuresSelectOptions
) {
    if (!optionsFile.empty()) {
        CB_ENSURE(NFs::Exists(optionsFile), "Params file does not exist " << optionsFile);
        TIFStream in(optionsFile);
        NJson::TJsonValue fromOptionsFile;
        CB_ENSURE(NJson::ReadJsonTree(&in, &fromOptionsFile), "can't parse params file");
        NCatboostOptions::PlainJsonToOptions(fromOptionsFile, catBoostJsonOptions, outputOptionsJson, featuresSelectOptions);
    }
    if (!outputOptionsJson->Has("train_dir")) {
        (*outputOptionsJson)["train_dir"] = ".";
    }
}

void CopyIgnoredFeaturesToPoolParams(
    const NJson::TJsonValue& catBoostJsonOptions,
    NCatboostOptions::TPoolLoadParams* poolLoadParams
) {
    poolLoadParams->IgnoredFeatures = GetOptionIgnoredFeatures(catBoostJsonOptions);
    const auto taskType = NCatboostOptions::GetTaskType(catBoostJsonOptions);
    poolLoadParams->Validate(taskType);
}

void BindQuantizerPoolLoadParams(NLastGetopt::TOpts* parser, NCatboostOptions::TPoolLoadParams* loadParamsPtr) {
    BindColumnarPoolFormatParams(parser, &(loadParamsPtr->ColumnarPoolFormatParams));
    parser->AddLongOption("input-borders-file", "file with borders")
            .RequiredArgument("PATH")
            .StoreResult(&loadParamsPtr->BordersFile);

    parser->AddLongOption("feature-names-path", "path to feature names data")
        .RequiredArgument("[SCHEME://]PATH")
        .Handler1T<TStringBuf>([loadParamsPtr](const TStringBuf& str) {
            loadParamsPtr->FeatureNamesPath = TPathWithScheme(str, "dsv");
        });
}

void BindPoolLoadParams(NLastGetopt::TOpts* parser, NCatboostOptions::TPoolLoadParams* loadParamsPtr) {
    BindQuantizerPoolLoadParams(parser, loadParamsPtr);

    parser->AddLongOption("cv-no-shuffle", "Do not shuffle dataset before cross-validation")
      .NoArgument()
      .Handler0([loadParamsPtr]() {
      loadParamsPtr->CvParams.Shuffle = false;
        });

    parser->AddLongOption('f', "learn-set", "learn set path")
        .RequiredArgument("[SCHEME://]PATH")
        .Handler1T<TStringBuf>([loadParamsPtr](const TStringBuf& str) {
            loadParamsPtr->LearnSetPath = TPathWithScheme(str, "dsv");
        });

    parser->AddLongOption('t', "test-set", "path to one or more test sets")
        .RequiredArgument("[SCHEME://]PATH[,[SCHEME://]PATH...]")
        .Handler1T<TStringBuf>([loadParamsPtr](const TStringBuf& str) {
            for (const auto& path : StringSplitter(str).Split(',').SkipEmpty()) {
                if (!path.empty()) {
                    loadParamsPtr->TestSetPaths.emplace_back(TString{path.Token()}, "dsv");
                }
            }
            CB_ENSURE(!loadParamsPtr->TestSetPaths.empty(), "Empty test path");
        });

    parser->AddLongOption("learn-pairs", "path to learn pairs")
        .RequiredArgument("[SCHEME://]PATH")
        .Handler1T<TStringBuf>([loadParamsPtr](const TStringBuf& str) {
            loadParamsPtr->PairsFilePath = TPathWithScheme(str, "dsv-flat");
        });

    parser->AddLongOption("test-pairs", "path to test pairs")
        .RequiredArgument("[SCHEME://]PATH")
        .Handler1T<TStringBuf>([loadParamsPtr](const TStringBuf& str) {
            loadParamsPtr->TestPairsFilePath = TPathWithScheme(str, "dsv-flat");
        });

    parser->AddLongOption("learn-graph", "path to learn dataset graph")
        .RequiredArgument("[SCHEME://]PATH")
        .Handler1T<TStringBuf>([loadParamsPtr](const TStringBuf& str) {
            loadParamsPtr->GraphFilePath = TPathWithScheme(str, "dsv-flat");
        })
        .Help("graph is used for calculating aggregation features");

    parser->AddLongOption("test-graph", "path to test dataset graph")
        .RequiredArgument("[SCHEME://]PATH")
        .Handler1T<TStringBuf>([loadParamsPtr](const TStringBuf& str) {
            loadParamsPtr->TestGraphFilePath = TPathWithScheme(str, "dsv-flat");
        })
        .Help("graph is used for calculating aggregation features");

    parser->AddLongOption("learn-group-weights", "path to learn group weights")
        .RequiredArgument("[SCHEME://]PATH")
        .Handler1T<TStringBuf>([loadParamsPtr](const TStringBuf& str) {
            loadParamsPtr->GroupWeightsFilePath = TPathWithScheme(str, "file");
        });

    parser->AddLongOption("test-group-weights", "path to test group weights")
        .RequiredArgument("[SCHEME://]PATH")
        .Handler1T<TStringBuf>([loadParamsPtr](const TStringBuf& str) {
            loadParamsPtr->TestGroupWeightsFilePath = TPathWithScheme(str, "file");
        });

    parser->AddLongOption("learn-timestamps", "path to learn timestamps")
        .RequiredArgument("[SCHEME://]PATH")
        .Handler1T<TStringBuf>([loadParamsPtr](const TStringBuf& str) {
            loadParamsPtr->TimestampsFilePath = TPathWithScheme(str, "file");
        });

    parser->AddLongOption("test-timestamps", "path to test timestamps")
        .RequiredArgument("[SCHEME://]PATH")
        .Handler1T<TStringBuf>([loadParamsPtr](const TStringBuf& str) {
            loadParamsPtr->TestTimestampsFilePath = TPathWithScheme(str, "file");
        });

    parser->AddLongOption("learn-baseline", "path to learn baseline")
        .RequiredArgument("[SCHEME://]PATH")
        .Handler1T<TStringBuf>([loadParamsPtr](const TStringBuf& str) {
            loadParamsPtr->BaselineFilePath = TPathWithScheme(str, "file");
        });

    parser->AddLongOption("test-baseline", "path to test baseline")
        .RequiredArgument("[SCHEME://]PATH")
        .Handler1T<TStringBuf>([loadParamsPtr](const TStringBuf& str) {
            loadParamsPtr->TestBaselineFilePath = TPathWithScheme(str, "file");
        });

    const auto cvDescription = TString::Join(
        "Cross validation type. Should be one of: ",
        GetEnumAllNames<ECrossValidation>(),
        ". Classical: test on fold n of k, n is 0-based",
        ". Inverted: train on fold n of k, n is 0-based",
        ". All cv types have two parameters n and k, they should be written in format cvtype:n;k.");
    parser->AddLongOption("cv", cvDescription)
        .RequiredArgument("string")
        .Handler1T<TString>([loadParamsPtr](const auto& str) {
            CB_ENSURE(
                !loadParamsPtr->CvParams.Initialized(),
                "Cross-validation params have already been initialized"
            );
            const auto cvType = FromString<ECrossValidation>(TStringBuf(str).Before(':'));
            const auto params = TStringBuf(str).After(':');
            if (cvType == ECrossValidation::Classical || cvType == ECrossValidation::Inverted ||
                cvType == ECrossValidation::TimeSeries) {
                Split(params, ';', loadParamsPtr->CvParams.FoldIdx, loadParamsPtr->CvParams.FoldCount);
                loadParamsPtr->CvParams.Type = cvType;
            }
            loadParamsPtr->CvParams.Check();
        });

    parser->AddLongOption("cv-rand", "cross-validation random seed")
        .RequiredArgument("seed")
        .StoreResult(&loadParamsPtr->CvParams.PartitionRandSeed);

    parser->AddLongOption(
       "hosts-already-contain-loaded-data",
       "[Distributed mode specific] Datasets' data has already been loaded to worker hosts,"
       "do not initiate its loading from master"
       )
       .NoArgument()
       .Handler0([loadParamsPtr]() {
            loadParamsPtr->HostsAlreadyContainLoadedData = true;
        });

    parser->AddLongOption("precomputed-data-meta", "file with precomputed data metadata")
        .RequiredArgument("PATH")
        .StoreResult(&loadParamsPtr->PrecomputedMetadataFile);

    parser->AddLongOption("pool-metainfo-path", "json file with pool metainfo")
        .RequiredArgument("[SCHEME://]PATH")
        .Handler1T<TStringBuf>([loadParamsPtr](const TStringBuf& str) {
            loadParamsPtr->PoolMetaInfoPath = TPathWithScheme(str);
        });
}

static void BindMetricParams(NLastGetopt::TOpts* parserPtr, NJson::TJsonValue* plainJsonPtr) {
    auto& parser = *parserPtr;
    const auto allObjectives = GetAllObjectives();
    const auto lossFunctionDescription = TString::Join(
        "Should be one of: ",
        JoinSeq(", ", allObjectives),
        ". A loss might have params, then params should be written in format Loss:paramName=value.");
    parser
        .AddLongOption("loss-function", lossFunctionDescription)
        .RequiredArgument("string")
        .Handler1T<TString>([plainJsonPtr, allObjectives](const auto& value) {
            const auto& lossFunctionName = ToString(TStringBuf(value).Before(':'));
            const auto enum_ = FromString<ELossFunction>(lossFunctionName);
            CB_ENSURE(IsIn(allObjectives, enum_), lossFunctionName + " objective is not known");
            (*plainJsonPtr)["loss_function"] = value;
        });

    const auto customMetricsDescription = TString::Join(
        "A metric might have parameters, then params should be written in format Loss:paramName=value. Loss should be one of: ",
        GetEnumAllNames<ELossFunction>());
    parser.AddLongOption("custom-metric", customMetricsDescription)
        .AddLongName("custom-loss")
        .RequiredArgument("comma separated list of metric functions")
        .Handler1T<TString>([plainJsonPtr](const TString& lossFunctionsLine) {
            for (const auto& lossFunction : StringSplitter(lossFunctionsLine).Split(',').SkipEmpty()) {
                FromString<ELossFunction>(lossFunction.Token().Before(':'));
                (*plainJsonPtr)["custom_metric"].AppendValue(NJson::TJsonValue(lossFunction.Token()));
            }
            CB_ENSURE(!(*plainJsonPtr)["custom_metric"].GetArray().empty(), "Empty custom metrics list " << lossFunctionsLine);
        });

    parser.AddLongOption("eval-metric")
        .RequiredArgument("string")
        .Handler1T<TString>([plainJsonPtr](const TString& metric) {
            (*plainJsonPtr)["eval_metric"] = metric;
        })
        .Help("evaluation metric for overfitting detector (if enabled) and best model "
              "selection in format MetricName:param=value. If not specified default metric for objective is used.");
}

static void BindOutputParams(NLastGetopt::TOpts* parserPtr, NJson::TJsonValue* plainJsonPtr) {
    auto& parser = *parserPtr;
    parser.AddLongOption('m', "model-file", "model file name")
        .RequiredArgument("PATH")
        .Handler1T<TString>([plainJsonPtr](const TString& name) {
            (*plainJsonPtr)["result_model_file"] = name;
            (*plainJsonPtr)["model_format"].AppendValue(ToString(NCatboostOptions::DefineModelFormat(name)));
        });

    parser.AddLongOption("model-format")
            .RequiredArgument("comma separated list of formats")
            .Handler1T<TString>([plainJsonPtr](const TString& formatsLine) {
                for (const auto& format : StringSplitter(formatsLine).Split(',').SkipEmpty()) {
                    const auto enum_ = FromString<EModelType>(format.Token());
                    (*plainJsonPtr)["model_format"].AppendValue(ToString(enum_));
                }
                CB_ENSURE(!(*plainJsonPtr)["model_format"].GetArray().empty(), "Empty model format list " << formatsLine);
            })
            .Help(BuildModelFormatHelpMessage() + " Corresponding extensions will be added to model-file if more than one format is set.");

    parser.AddLongOption("eval-file", "eval output file name")
        .RequiredArgument("PATH")
        .Handler1T<TString>([plainJsonPtr](const TString& name) {
            (*plainJsonPtr)["eval_file_name"] = name;
        });

    parser.AddLongOption("output-borders-file", "float feature borders output file name")
            .RequiredArgument("PATH")
            .Handler1T<TString>([plainJsonPtr](const TString& name) {
                (*plainJsonPtr)["output_borders"] = name;
            });

    parser.AddLongOption("fstr-file", "Save fstr to this file")
        .RequiredArgument("filename")
        .Handler1T<TString>([plainJsonPtr](const TString& name) {
            (*plainJsonPtr)["fstr_regular_file"] = name;
        });

    const auto customFstrTypeDescription = TString::Join(
            "Should be one of: ", GetEnumAllNames<EFstrCalculatedInFitType>());
    parser.AddLongOption("fstr-type", customFstrTypeDescription)
        .RequiredArgument("fstr-type")
        .Handler1T<TString>([plainJsonPtr](const TString& name) {
            (*plainJsonPtr)["fstr_type"] = name;
        });

    parser.AddLongOption("fstr-internal-file", "Save internal fstr values to this file")
        .RequiredArgument("filename")
        .Handler1T<TString>([plainJsonPtr](const TString& name) {
            (*plainJsonPtr)["fstr_internal_file"] = name;
        });

    parser.AddLongOption("training-options-file", "Save training options to this file")
        .RequiredArgument("filename")
        .Handler1T<TString>([plainJsonPtr](const TString& name) {
            (*plainJsonPtr)["training_options_file"] = name;
        });

    parser.AddLongOption("learn-err-log", "file to log error function on train")
        .RequiredArgument("file")
        .Handler1T<TString>([plainJsonPtr](const TString& name) {
            (*plainJsonPtr)["learn_error_log"] = name;
        });

    parser.AddLongOption("test-err-log", "file to log error function on test")
        .RequiredArgument("file")
        .Handler1T<TString>([plainJsonPtr](const TString& name) {
            (*plainJsonPtr)["test_error_log"] = name;
        });

    parser.AddLongOption("json-log", "json to log metrics and time information")
        .RequiredArgument("file")
        .Handler1T<TString>([plainJsonPtr](const TString& name) {
            (*plainJsonPtr)["json_log"] = name;
        });

    parser.AddLongOption("profile-log", "file to log profile information")
        .RequiredArgument("file")
        .Handler1T<TString>([plainJsonPtr](const TString& name) {
            (*plainJsonPtr)["profile_log"] = name;
        });

    parser.AddLongOption("trace-log", "path for trace log")
        .RequiredArgument("file")
        .Handler1T<TString>([](const TString& name) {
            TCatBoostLogSettings::GetRef().Log.ResetTraceBackend(CreateLogBackend(name));
        });

    parser.AddLongOption("use-best-model", "If true - save all trees until best iteration on test.")
        .OptionalValue("true", "bool")
        .Handler1T<TString>([plainJsonPtr](const TString& useBestModel) {
            (*plainJsonPtr)["use_best_model"] = FromString<bool>(useBestModel);
        });

    parser.AddLongOption("best-model-min-trees", "Minimal number of trees the best model should have.")
        .RequiredArgument("int")
        .Handler1T<int>([plainJsonPtr](const auto bestModelMinTrees) {
            (*plainJsonPtr)["best_model_min_trees"] = bestModelMinTrees;
        });

    parser.AddLongOption("name", "name to be displayed in visualizator")
        .RequiredArgument("name")
        .Handler1T<TString>([plainJsonPtr](const TString& name) {
            (*plainJsonPtr)["name"] = name;
        });

    parser.AddLongOption("train-dir", "directory to store train logs")
        .RequiredArgument("PATH")
        .Handler1T<TString>([plainJsonPtr](const TString& path) {
            (*plainJsonPtr)["train_dir"] = path;
        });

    parser.AddLongOption("verbose", "period of printing metrics to stdout; never if 0 or False")
        .RequiredArgument("PERIOD")
        .Handler1T<TString>([plainJsonPtr](const TString& period) {
            try {
                (*plainJsonPtr)["verbose"] = FromString<int>(period);
            } catch (...) {
                (*plainJsonPtr)["verbose"] = int(FromString<bool>(period));
            }
        });

    parser.AddLongOption("metric-period", "period of calculating metrics")
        .RequiredArgument("int")
        .Handler1T<int>([plainJsonPtr](const auto period) {
        (*plainJsonPtr)["metric_period"] = period;
    });

    parser.AddLongOption("snapshot-file", "use progress file for restoring progress after crashes")
        .RequiredArgument("PATH")
        .Handler1T<TString>([plainJsonPtr](const TString& path) {
            (*plainJsonPtr)["save_snapshot"] = true;
            (*plainJsonPtr)["snapshot_file"] = path;
        });

    parser.AddLongOption("snapshot-interval", "interval between saving snapshots (seconds)")
        .RequiredArgument("int")
        .Handler1T<TString>([plainJsonPtr](const TString& interval) {
            (*plainJsonPtr)["snapshot_interval"] = FromString<int>(interval);
        });

    parser.AddLongOption("output-columns")
            .RequiredArgument("Comma separated list of column indexes")
            .Handler1T<TString>([plainJsonPtr](const TString& indexesLine) {
                (*plainJsonPtr)["output_columns"] = NULL;
                for (const auto& t : StringSplitter(indexesLine).Split(',').SkipEmpty()) {
                    (*plainJsonPtr)["output_columns"].AppendValue(t.Token());
                }
                CB_ENSURE(!(*plainJsonPtr)["output_columns"].GetArray().empty(), "Empty column indexes list " << indexesLine);
            });

    parser.AddLongOption("prediction-type")
        .RequiredArgument("Comma separated list of prediction types. Every prediction type should be one of: Probability, Class, RawFormulaVal. CPU only")
        .Handler1T<TString>([plainJsonPtr](const TString& predictionTypes) {
            for (const auto& t : StringSplitter(predictionTypes).Split(',').SkipEmpty()) {
                (*plainJsonPtr)["prediction_type"].AppendValue(t.Token());
            }
            CB_ENSURE(!(*plainJsonPtr)["prediction_type"].GetArray().empty(), "Empty prediction type list " << predictionTypes);
        });
    parser.AddLongOption("final-ctr-computation-mode", "Should be one of: Default, Skip. Use all pools to compute final ctrs by default or skip final ctr computation. WARNING: model can't be applied if final ctrs computation is skipped!")
            .RequiredArgument("string")
            .Handler1T<TString>([plainJsonPtr](const TString& finalCtrComputationMode) {
                (*plainJsonPtr)["final_ctr_computation_mode"] = finalCtrComputationMode;
            });
    parser.AddLongOption("allow-writing-files", "Allow writing files on disc. Possible values: true, false")
            .RequiredArgument("bool")
            .Handler1T<TString>([plainJsonPtr](const TString& param) {
                (*plainJsonPtr)["allow_writing_files"] = FromString<bool>(param);
            });

}

static void BindBoostingParams(NLastGetopt::TOpts* parserPtr, NJson::TJsonValue* plainJsonPtr) {
    auto& parser = *parserPtr;
    parser.AddLongOption('i', "iterations", "iterations count")
        .RequiredArgument("ITERATIONS")
        .Handler1T<int>([plainJsonPtr](int iterations) {
            (*plainJsonPtr)["iterations"] = iterations;
        });

    parser.AddLongOption('w', "learning-rate", "learning rate")
        .RequiredArgument("float")
        .Handler1T<float>([plainJsonPtr](float w) {
            (*plainJsonPtr)["learning_rate"] = w;
        });

    parser.AddLongOption("fold-len-multiplier", "Fold length multiplier. Should be greater than 1")
        .RequiredArgument("float")
        .Handler1T<float>([plainJsonPtr](float multiplier) {
            (*plainJsonPtr)["fold_len_multiplier"] = multiplier;
        });

    parser.AddLongOption("approx-on-full-history")
        .NoArgument()
        .Handler0([plainJsonPtr]() {
            (*plainJsonPtr)["approx_on_full_history"] = true;
        })
        .Help("Use full history to calculate approxes.");

    parser.AddLongOption("fold-permutation-block",
                         "Enables fold permutation by blocks of given length, preserving documents order inside each block.")
        .RequiredArgument("BLOCKSIZE")
        .Handler1T<int>([plainJsonPtr](int blockSize) {
            (*plainJsonPtr)["fold_permutation_block"] = blockSize;
        });

    parser
        .AddLongOption("min-fold-size")
        .RequiredArgument("INT")
        .Help("Sets suggested min fold size")
        .Handler1T<int>([plainJsonPtr](int blockSize) {
            (*plainJsonPtr)["min_fold_size"] = blockSize;
        });

    parser
        .AddLongOption('p', "permutations")
        .RequiredArgument("INT")
        .Help("GPU only. Sets permutation count. CatBoost use 3 learning permutations and 1 estimation by default. Lower values will decrease learning time, but could affect quality.")
        .Handler1T<int>([plainJsonPtr](int blockSize) {
            (*plainJsonPtr)["permutation_count"] = blockSize;
        });

    parser
        .AddLongOption("boost-from-average",
                       "Enables to initialize approx values by best constant value for specified loss function. \
                       Available for RMSE, Logloss, CrossEntropy, Quantile and MAE. Possible values: true, false.")
        .RequiredArgument("bool")
        .Handler1T<TString>([plainJsonPtr](const TString& param) {
            (*plainJsonPtr)["boost_from_average"] = FromString<bool>(param);
        });

    const auto boostingTypeHelp = TString::Join(
        "Set boosting type (",
        GetEnumAllNames<EBoostingType>(),
        ") By default CatBoost use dynamic-boosting scheme. For best performance you could set it to ",
        ToString(EBoostingType::Plain));
    parser
        .AddLongOption("boosting-type")
        .RequiredArgument("BoostingType")
        .Help(boostingTypeHelp)
        .Handler1T<EBoostingType>([plainJsonPtr](const auto boostingType) {
            (*plainJsonPtr)["boosting_type"] = ToString(boostingType);
        });
    const auto dataPartitionHelp = TString::Join(
        "Sets method to split learn samples between multiple workers (GPU only currently). Possible values are: ",
        GetEnumAllNames<EDataPartitionType>(),
        ". Default depends on learning mode and dataset.");
    parser
        .AddLongOption("data-partition")
        .RequiredArgument("PartitionType")
        .Help(dataPartitionHelp)
        .Handler1T<EDataPartitionType>([plainJsonPtr](const auto type) {
            (*plainJsonPtr)["data_partition"] = ToString(type);
        });

    parser.AddLongOption("od-pval",
                         "pValue threshold for overfitting detector. For good results use threshold in [1e-10, 1e-2]."
                         "Specified test-set is required.")
        .RequiredArgument("float")
        .Handler1T<float>([plainJsonPtr](float pval) {
            (*plainJsonPtr)["od_pval"] = pval;
        });

    parser.AddLongOption("od-wait",
                         "number of iterations which overfitting detector will wait after new best error")
        .RequiredArgument("int")
        .Handler1T<int>([plainJsonPtr](int iters) {
            (*plainJsonPtr)["od_wait"] = iters;
        });

    parser.AddLongOption("od-type", "Should be one of {IncToDec, Iter}")
        .RequiredArgument("detector-type")
        .Handler1T<EOverfittingDetectorType>([plainJsonPtr](const auto type) {
            (*plainJsonPtr)["od_type"] = ToString(type);
        });

    parser.AddLongOption("model-shrink-rate",
                         "This parameter enables shrinkage of model at the start of each iteration. CPU only."
                         "For Constant mode shrinkage coefficient is calculated as (1 - model_shrink_rate * learning_rate)."
                         "For Decreasing mode shrinkage coefficient is calculated as (1 - model_shrink_rate / iteration)."
                         "Shrinkage coefficient should be in [0, 1).")
        .RequiredArgument("float")
        .Handler1T<float>([plainJsonPtr](float modelShrinkRate) {
            (*plainJsonPtr)["model_shrink_rate"] = modelShrinkRate;
        });

    parser.AddLongOption("model-shrink-mode", "Mode of shrink coefficient calculation. Possible values: Constant, Decreasing.")
        .RequiredArgument("ShrinkMode")
        .Handler1T<EModelShrinkMode>([plainJsonPtr](EModelShrinkMode modelShrinkMode) {
            (*plainJsonPtr)["model_shrink_mode"] = ToString(modelShrinkMode);
        });

    parser
        .AddLongOption("langevin")
        .RequiredArgument("bool")
        .Help("Enables the Stochastic Gradient Langevin Boosting.")
        .Handler1T<TString>([plainJsonPtr](const TString& isEnabled) {
            (*plainJsonPtr)["langevin"] = FromString<bool>(isEnabled);
        });

    parser
        .AddLongOption("posterior-sampling")
        .RequiredArgument("bool")
        .Help("Enables the posterior sampling.")
        .Handler1T<TString>([plainJsonPtr](const TString& isEnabled) {
            (*plainJsonPtr)["posterior_sampling"] = FromString<bool>(isEnabled);
        });


    parser.AddLongOption("diffusion-temperature", "Langevin boosting diffusion temperature.")
        .RequiredArgument("float")
        .Handler1T<float>([plainJsonPtr](float diffusionTemperature) {
            (*plainJsonPtr)["diffusion_temperature"] = diffusionTemperature;
        });
}

static void BindModelBasedEvalParams(NLastGetopt::TOpts* parserPtr, NJson::TJsonValue* plainJsonPtr) {
    auto& parser = *parserPtr;
    parser
        .AddLongOption("features-to-evaluate")
        .RequiredArgument("INDEXES[;INDEXES...]")
        .Help("Evaluate impact of each set of features on test error; each set is a comma-separated list of indices and index intervals, e.g. 4,78-89,312.")
        .Handler1T<TString>([plainJsonPtr](const TString& indicesLine) {
            (*plainJsonPtr)["features_to_evaluate"] = indicesLine;
        });
    parser
        .AddLongOption("baseline-model-snapshot")
        .RequiredArgument("PATH")
        .Help("Snapshot of base model training.")
        .Handler1T<TString>([plainJsonPtr](const TString& trainingPath) {
            (*plainJsonPtr)["baseline_model_snapshot"] = trainingPath;
        });
    parser
        .AddLongOption("offset")
        .RequiredArgument("INT")
        .Help("Evaluate using this number of last iterations of the base model training.")
        .Handler1T<int>([plainJsonPtr](int offset) {
            (*plainJsonPtr)["offset"] = offset;
        });
    parser
        .AddLongOption("experiment-count")
        .RequiredArgument("INT")
        .Help("Number of experiments for model-based evaluation.")
        .Handler1T<int>([plainJsonPtr](int experimentCount) {
            (*plainJsonPtr)["experiment_count"] = experimentCount;
        });
    parser
        .AddLongOption("experiment-size")
        .RequiredArgument("INT")
        .Help("Number of iterations in one experiment.")
        .Handler1T<int>([plainJsonPtr](int experimentSize) {
            (*plainJsonPtr)["experiment_size"] = experimentSize;
        });
    parser
        .AddLongOption("use-evaluated-features-in-baseline-model")
        .NoArgument()
        .Help("Use all evaluated features in baseline model rather than zero out them.")
        .Handler0([plainJsonPtr]() {
            (*plainJsonPtr)["use_evaluated_features_in_baseline_model"] = true;
        });
}

static void BindFeatureEvalParams(NLastGetopt::TOpts* parserPtr, NJson::TJsonValue* plainJsonPtr) {
    auto& parser = *parserPtr;
    parser
        .AddLongOption("features-to-evaluate")
        .RequiredArgument("INDEXES[;INDEXES...]")
        .Help("Evaluate impact of each set of features on test error; each set is a comma-separated list of indices and index intervals, e.g. 4,78-89,312.")
        .Handler1T<TString>([plainJsonPtr](const TString& indicesLine) {
            (*plainJsonPtr)["features_to_evaluate"] = indicesLine;
        });
    parser
        .AddLongOption("feature-eval-mode")
        .RequiredArgument("STRING")
        .Help("Feature evaluation mode; must be one of " + GetEnumAllNames<NCB::EFeatureEvalMode>())
        .Handler1T<NCB::EFeatureEvalMode>([plainJsonPtr](const auto mode) {
            (*plainJsonPtr)["feature_eval_mode"] = ToString(mode);
        });
    parser
        .AddLongOption("feature-eval-output-file")
        .RequiredArgument("STRING")
        .Help("file containing feature evaluation summary (p-values and metric deltas for each set of tested features)")
        .Handler1T<TString>([plainJsonPtr](const auto filename) {
            (*plainJsonPtr)["eval_feature_file"] = ToString(filename);
        });
    parser
        .AddLongOption("processors-usage-output-file")
        .RequiredArgument("STRING")
        .Help("file containing processors usage summary (time in seconds and number of iterations for each processors; collected on GPU only)")
        .Handler1T<TString>([plainJsonPtr](const auto filename) {
            (*plainJsonPtr)["processors_usage_file"] = ToString(filename);
        });
    parser
        .AddLongOption("offset")
        .RequiredArgument("INT")
        .Help("First fold for feature evaluation")
        .Handler1T<ui32>([plainJsonPtr](const auto offset) {
            (*plainJsonPtr)["offset"] = offset;
        });
    parser
        .AddLongOption("fold-count")
        .RequiredArgument("INT")
        .Help("Fold count for feature evaluation")
        .Handler1T<ui32>([plainJsonPtr](const auto foldCount) {
            (*plainJsonPtr)["fold_count"] = foldCount;
        });
    parser
        .AddLongOption("fold-size-unit")
        .RequiredArgument("STRING")
        .Help("Units to specify fold size for feature evaluation; must be one of " + GetEnumAllNames<ESamplingUnit>())
        .Handler1T<ESamplingUnit>([plainJsonPtr](const auto foldSizeUnit) {
            (*plainJsonPtr)["fold_size_unit"] = ToString(foldSizeUnit);
        });
    parser
        .AddLongOption("fold-size")
        .RequiredArgument("INT")
        .Help("Fold size for feature evaluation; number of fold-size-units")
        .Handler1T<int>([plainJsonPtr](const auto foldSize) {
            (*plainJsonPtr)["fold_size"] = foldSize;
            CB_ENSURE(!plainJsonPtr->Has("relative_fold_size"), "Fold size and relative fold size are mutually exclusive");
        });
    parser
        .AddLongOption("relative-fold-size")
        .RequiredArgument("float")
        .Help("Relative fold size for feature evaluation; fraction of total number of fold-size-units in dataset")
        .Handler1T<float>([plainJsonPtr](const auto foldSize) {
            (*plainJsonPtr)["relative_fold_size"] = foldSize;
            CB_ENSURE(!plainJsonPtr->Has("fold_size"), "Fold size and relative fold size are mutually exclusive");
        });
    parser
        .AddLongOption("timesplit-quantile")
        .RequiredArgument("float")
        .Help("Quantile for timesplit in feature evaluation")
        .Handler1T<float>([plainJsonPtr](const auto quantile) {
            (*plainJsonPtr)["timesplit_quantile"] = quantile;
        });
}

static void BindFeaturesSelectParams(NLastGetopt::TOpts* parserPtr, NJson::TJsonValue* plainJsonPtr) {
    auto& parser = *parserPtr;
    parser
        .AddLongOption("features-for-select")
        .RequiredArgument("INDEX,INDEX-INDEX,...")
        .Help("From which features perform selection; each set is a comma-separated list of indices and index intervals, e.g. 4,78-89,312.")
        .Handler1T<TString>([plainJsonPtr](const TString& indicesLine) {
            (*plainJsonPtr)["features_for_select"] = indicesLine;
        });
    parser
        .AddLongOption("num-features-to-select")
        .RequiredArgument("int")
        .Help("How many features to select from features-for-select.")
        .Handler1T<int>([plainJsonPtr](const int numberOfFeaturesToSelect) {
            (*plainJsonPtr)["num_features_to_select"] = numberOfFeaturesToSelect;
        });
    parser
        .AddLongOption("features-tags-for-select")
        .RequiredArgument("TAG,TAG,...")
        .Help("From which features tags perform selection.")
        .Handler1T<TString>([plainJsonPtr](const TString& tagNamesLine) {
            for (const auto& tag : StringSplitter(tagNamesLine).Split(',').SkipEmpty()) {
                (*plainJsonPtr)["features_tags_for_select"].AppendValue(TStringBuf(tag));
            }
            CB_ENSURE(
                !(*plainJsonPtr)["features_tags_for_select"].GetArray().empty(),
                "Empty features tags for selection list " << tagNamesLine
            );
        });
    parser
        .AddLongOption("num-features-tags-to-select")
        .RequiredArgument("int")
        .Help("How many features tags to select from features-tags-for-select.")
        .Handler1T<int>([plainJsonPtr](const int numberOfFeaturesTagsToSelect) {
            (*plainJsonPtr)["num_features_tags_to_select"] = numberOfFeaturesTagsToSelect;
        });
    parser
        .AddLongOption("features-selection-steps")
        .RequiredArgument("int")
        .Help("How many steps to perform during feature selection.")
        .Handler1T<int>([plainJsonPtr](const int steps) {
            (*plainJsonPtr)["features_selection_steps"] = steps;
        });
    parser
        .AddLongOption("train-final-model")
        .NoArgument()
        .Help("Need to train and save model after features selection.")
        .Handler0([plainJsonPtr]() {
            (*plainJsonPtr)["train_final_model"] = true;
        });
    parser
        .AddLongOption("features-selection-result-path")
        .RequiredArgument("PATH")
        .Help("Where to save results of features selection.")
        .Handler1T<TString>([plainJsonPtr](const TString& path) {
            (*plainJsonPtr)["features_selection_result_path"] = path;
        });
    parser
        .AddLongOption("features-selection-algorithm")
        .Help(TString::Join(
            "Which algorithm to use for features selection.\n",
            "Should be one of: ", GetEnumAllNames<EFeaturesSelectionAlgorithm>()))
        .Handler1T<EFeaturesSelectionAlgorithm>([plainJsonPtr](const auto algorithm) {
            (*plainJsonPtr)["features_selection_algorithm"] = ToString(algorithm);
        });
    parser
        .AddLongOption("features-selection-grouping")
        .Help(TString::Join(
            "Which grouping to use for features selection.\n",
            "Should be one of: ", GetEnumAllNames<EFeaturesSelectionGrouping>()))
        .Handler1T<EFeaturesSelectionGrouping>([plainJsonPtr](const auto grouping) {
            (*plainJsonPtr)["features_selection_grouping"] = ToString(grouping);
        });
    parser
        .AddLongOption("shap-calc-type")
        .DefaultValue("Regular")
        .Help("Should be one of: 'Approximate', 'Regular', 'Exact'.")
        .Handler1T<ECalcTypeShapValues>([plainJsonPtr](const ECalcTypeShapValues calcType) {
            (*plainJsonPtr)["shap_calc_type"] = ToString(calcType);
        });
}

static void BindTreeParams(NLastGetopt::TOpts* parserPtr, NJson::TJsonValue* plainJsonPtr) {
    auto& parser = *parserPtr;
    parser.AddLongOption("rsm", "random subspace method (feature bagging)")
        .RequiredArgument("float")
        .Handler1T<float>([plainJsonPtr](float rsm) {
            (*plainJsonPtr)["rsm"] = rsm;
        });

    parser.AddLongOption("leaf-estimation-iterations", "gradient iterations count")
        .RequiredArgument("int")
        .Handler1T<int>([plainJsonPtr](int iterations) {
            (*plainJsonPtr)["leaf_estimation_iterations"] = iterations;
        });

    const auto leafEstimationBacktrackingHelp = TString::Join(
        "Backtracking type; Must be one of: ",
        GetEnumAllNames<ELeavesEstimationStepBacktracking>());
    parser.AddLongOption("leaf-estimation-backtracking", leafEstimationBacktrackingHelp)
        .RequiredArgument("str")
        .Handler1T<ELeavesEstimationStepBacktracking>([plainJsonPtr](const auto type) {
            (*plainJsonPtr)["leaf_estimation_backtracking"] = ToString(type);
        });

    parser.AddLongOption('n', "depth", "tree depth")
        .RequiredArgument("int")
        .Handler1T<int>([plainJsonPtr](int depth) {
            (*plainJsonPtr)["depth"] = depth;
        });


    parser.AddLongOption("grow-policy", "Tree growing policy. Must be one of: " + GetEnumAllNames<EGrowPolicy>())
        .RequiredArgument("type")
        .Handler1T<TString>([plainJsonPtr](const TString& policy) {
            (*plainJsonPtr)["grow_policy"] = policy;
        });

    parser.AddLongOption("max-leaves", "Maximum number of leaves per tree")
        .RequiredArgument("INT")
        .Handler1T<ui32>([plainJsonPtr](const ui32 maxLeaves) {
            (*plainJsonPtr)["max_leaves"] = maxLeaves;
        });

    parser.AddLongOption("min-data-in-leaf", "Minimum number of samples in leaf")
        .RequiredArgument("Double")
        .Handler1T<double>([plainJsonPtr](double minSamples) {
            (*plainJsonPtr)["min_data_in_leaf"] = minSamples;
        });

    parser.AddLongOption("l2-leaf-reg", "Regularization value. Should be >= 0")
        .RequiredArgument("float")
        .Handler1T<float>([plainJsonPtr](float reg) {
            (*plainJsonPtr)["l2_leaf_reg"] = reg;
        });

    parser.AddLongOption("meta-l2-leaf-exponent", "GPU only. Exponent value for meta L2 score function.")
        .RequiredArgument("float")
        .Handler1T<float>([plainJsonPtr](float exponent) {
            (*plainJsonPtr)["meta_l2_exponent"] = exponent;
        });

    parser.AddLongOption("meta-l2-leaf-frequency", "GPU only. Frequency value for meta L2 score function.")
        .RequiredArgument("float")
        .Handler1T<float>([plainJsonPtr](float frequency) {
            (*plainJsonPtr)["meta_l2_frequency"] = frequency;
        });

    parser.AddLongOption(
        "fixed-binary-splits",
        "GPU only. Binary features to put at the root of each tree. Colon-separated list of feature names, indices, or inclusive intervals of indices, e.g. 4:78-89:312")
        .RequiredArgument("INDICES or NAMES")
        .Handler1T<TString>([plainJsonPtr](const TString& indicesLine) {
            for (const auto& ignoredFeature : StringSplitter(indicesLine).Split(':')) {
                (*plainJsonPtr)["fixed_binary_splits"].AppendValue(ignoredFeature.Token());
            }
        });

    parser.AddLongOption("bayesian-matrix-reg", "Regularization value. Should be >= 0")
            .RequiredArgument("float")
            .Handler1T<float>([plainJsonPtr](float reg) {
                (*plainJsonPtr)["bayesian_matrix_reg"] = reg;
            });


    parser.AddLongOption("model-size-reg", "Model size regularization coefficient. Should be >= 0")
            .RequiredArgument("float")
            .Handler1T<float>([plainJsonPtr](float reg) {
                (*plainJsonPtr)["model_size_reg"] = reg;
            });

    parser.AddLongOption("dev-score-calc-obj-block-size",
                         "CPU only. Size of block of samples in score calculation. Should be > 0"
                         "Used only for learning speed tuning."
                         "Changing this parameter can affect results"
                         " due to numerical accuracy differences")
            .RequiredArgument("INT")
            .Handler1T<int>([plainJsonPtr](int size) {
                (*plainJsonPtr)["dev_score_calc_obj_block_size"] = size;
            });

    parser.AddLongOption("dev-efb-max-buckets",
                         "CPU only. Maximum bucket count in exclusive features bundle. "
                         "Should be in an integer between 0 and 65536. "
                         "Used only for learning speed tuning.")
            .RequiredArgument("INT")
            .Handler1T<int>([plainJsonPtr](int maxBuckets) {
                (*plainJsonPtr)["dev_efb_max_buckets"] = maxBuckets;
            });

    parser.AddLongOption("sparse-features-conflict-fraction",
                         "CPU only. Maximum allowed fraction of conflicting non-default values for features in exclusive features bundle."
                         "Should be a real value in [0, 1) interval.")
            .RequiredArgument("float")
            .Handler1T<float>([plainJsonPtr](float fraction) {
                (*plainJsonPtr)["sparse_features_conflict_fraction"] = fraction;
            });

    parser.AddLongOption("random-strength")
        .RequiredArgument("float")
        .Handler1T<float>([plainJsonPtr](float randomStrength) {
            (*plainJsonPtr)["random_strength"] = randomStrength;
        })
        .Help("score standard deviation multiplier");

    const auto leafEstimationMethodHelp = TString::Join(
        "Must be one of: ",
        GetEnumAllNames<ELeavesEstimation>());
    parser.AddLongOption("leaf-estimation-method", leafEstimationMethodHelp)
        .RequiredArgument("method-name")
        .Handler1T<ELeavesEstimation>([plainJsonPtr](const auto method) {
            (*plainJsonPtr)["leaf_estimation_method"] = ToString(method);
        });

    const auto scoreFunctionHelp = TString::Join(
        "Could be change during GPU learning only. Change score function to use. ",
        " Must be one of: ",
        GetEnumAllNames<EScoreFunction>());
    parser
        .AddLongOption("score-function")
        .RequiredArgument("STRING")
        .Help(scoreFunctionHelp)
        .Handler1T<EScoreFunction>([plainJsonPtr](const auto func) {
            (*plainJsonPtr)["score_function"] = ToString(func);
        });

    parser
        .AddLongOption("fold-size-loss-normalization")
        .RequiredArgument("FLAG")
        .Help("GPU only. Use fold size as loss normalizations for different fold models.")
        .NoArgument()
        .Handler0([plainJsonPtr]() {
            (*plainJsonPtr)["fold_size_loss_normalization"] = true;
        });

    parser
        .AddLongOption("add-ridge-penalty-for-loss-function")
        .RequiredArgument("FLAG")
        .Help("False by default. Could be changed on GPU only. Add ridge (l2) penalty for loss function for leaves estimation.")
        .NoArgument()
        .Handler0([plainJsonPtr]() {
            (*plainJsonPtr)["add_ridge_penalty_to_loss_function"] = true;
        });

    parser
        .AddLongOption("dev-max-ctr-complexity-for-border-cache")
        .RequiredArgument("FLAG")
        .Help("False by default. GPU only. Set max ctr complexity for which borders will be cached during learning.")
        .Handler1T<int>([plainJsonPtr](int limit) {
            (*plainJsonPtr)["dev_max_ctr_complexity_for_borders_cache"] = limit;
        });

    const auto bootstrapTypeHelp = TString::Join(
        "Bootstrap type. Change default way of sampling documents weights. Must be one of: ",
        GetEnumAllNames<EBootstrapType>(),
        ". By default CatBoost uses Bayesian for GPU and MVS for CPU.");
    parser
        .AddLongOption("bootstrap-type")
        .RequiredArgument("STRING")
        .Help(bootstrapTypeHelp)
        .Handler1T<EBootstrapType>([plainJsonPtr](const auto type) {
            (*plainJsonPtr)["bootstrap_type"] = ToString(type);
        });

    parser
        .AddLongOption("sampling-unit")
        .RequiredArgument("STRING")
        .Help("Allows to manage the sampling scheme. Sample weights for each object individually or for an entire group of objects together.")
        .Handler1T<ESamplingUnit>([plainJsonPtr](const auto type) {
            (*plainJsonPtr)["sampling_unit"] = ToString(type);
        });

    parser.AddLongOption("bagging-temperature")
        .AddLongName("tmp")
        .RequiredArgument("float")
        .Handler1T<float>([plainJsonPtr](float baggingTemperature) {
            (*plainJsonPtr)["bagging_temperature"] = baggingTemperature;
        })
        .Help("Controls intensity of Bayesian bagging. The higher the temperature the more aggressive bagging is. Typical values are in range [0, 1] (0 - no bagging, 1 - default). Available for Bayesian bootstap only");

    const auto samplingFrequencyHelp = TString::Join(
        "Controls how frequently to sample weights and objects when constructing trees. "
        "Possible values are ",
        GetEnumAllNames<ESamplingFrequency>());
    parser.AddLongOption("sampling-frequency")
        .RequiredArgument("string")
        .Handler1T<ESamplingFrequency>([plainJsonPtr](const ESamplingFrequency target) {
            (*plainJsonPtr)["sampling_frequency"] = ToString(target);
        })
        .Help(samplingFrequencyHelp);

    parser
        .AddLongOption("subsample")
        .RequiredArgument("Float")
        .Handler1T<float>([plainJsonPtr](float rate) {
            (*plainJsonPtr)["subsample"] = rate;
        })
        .Help("Controls sample rate for bagging. Could be used if bootstrap-type is Poisson, Bernoulli or MVS. \
            Possible values are from (0, 1]; 0.66 by default for Bernoulli and Poisson, 0.8 by default for MVS."
        );

    parser
        .AddLongOption("mvs-reg")
        .RequiredArgument("Float")
        .Handler1T<float>([plainJsonPtr](float mvs_reg) {
            (*plainJsonPtr)["mvs_reg"] = mvs_reg;
        })
        .Help("Controls the weight of denominator in MVS procedure.");

    parser
        .AddLongOption("observations-to-bootstrap")
        .RequiredArgument("FLAG")
        .Help("GPU only.Use bootstraped weights on learn and test folds. By default bootstrap used only for test fold part.")
        .Handler1T<TString>([plainJsonPtr](const TString& type) {
            (*plainJsonPtr)["observations_to_bootstrap"] = type;
        });

    parser
        .AddLongOption("monotone-constraints")
        .RequiredArgument("String")
        .Help("Monotone constraints for features. Possible formats: \"(1,0,0,-1)\" or \"0:1,3:-1\" or \"FeatureName1:1,FeatureName2:-1\"")
        .Handler1T<TString>([plainJsonPtr](const TString& monotoneConstraints) {
            (*plainJsonPtr)["monotone_constraints"] = monotoneConstraints;
        });

    parser
        .AddLongOption("dev-leafwise-approxes", "Calculate approxes independently in each leaf")
        .NoArgument()
        .Handler0([plainJsonPtr]() {
            (*plainJsonPtr)["dev_leafwise_approxes"] = true;
        });

    parser
        .AddLongOption("feature-weights")
        .RequiredArgument("String")
        .Help("Weights to multiply splits gain where specific feature is used. Possible formats: \"(1,0.5,10,1)\" or \"1:0.5,2:10\" or \"FeatureName1:0.5,FeatureName2:10\". Should be nonnegative.")
        .Handler1T<TString>([plainJsonPtr](const TString& featureWeights) {
            (*plainJsonPtr)["feature_weights"] = featureWeights;
        });

    parser
        .AddLongOption("penalties-coefficient")
        .RequiredArgument("Float")
        .Help("Common coefficient for feature penalties. 1 by default. Should be nonnegative.")
        .Handler1T<float>([plainJsonPtr](const float penaltiesCoefficient) {
            (*plainJsonPtr)["penalties_coefficient"] = penaltiesCoefficient;
        });

    parser
        .AddLongOption("first-feature-use-penalties")
        .RequiredArgument("String")
        .Help("Penalties for first use of feature in model. Possible formats: \"(0,0.5,10,0)\" or \"1:0.5,2:10\" or \"FeatureName1:0.5,FeatureName2:10\" Should be nonnegative.")
        .Handler1T<TString>([plainJsonPtr](const TString& firstFeatureUsePenalty) {
            (*plainJsonPtr)["first_feature_use_penalties"] = firstFeatureUsePenalty;
        });

    parser
        .AddLongOption("per-object-feature-penalties")
        .RequiredArgument("String")
        .Help("Penalties for first use of feature for each object in model. Possible formats: \"(0,0.5,10,0)\" or \"1:0.5,2:10\" or \"FeatureName1:0.5,FeatureName2:10\" Should be nonnegative.")
        .Handler1T<TString>([plainJsonPtr](const TString& perObjectFeaturePenalty) {
            (*plainJsonPtr)["per_object_feature_penalties"] = perObjectFeaturePenalty;
        });
}

static void BindCatFeatureParams(NLastGetopt::TOpts* parserPtr, NJson::TJsonValue* plainJsonPtr) {
    auto& parser = *parserPtr;
    parser.AddLongOption("max-ctr-complexity", "max count of cat features for combinations ctr")
        .RequiredArgument("int")
        .Handler1T<int>([plainJsonPtr](int count) {
            (*plainJsonPtr).InsertValue("max_ctr_complexity", count);
        });

    parser.AddLongOption("simple-ctr",
                         "Ctr description should be written in format CtrType[:TargetBorderCount=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num/denum]. CtrType should be one of: Borders, Buckets, BinarizedTargetMeanValue, Counter. TargetBorderType should be one of: Median, GreedyLogSum, UniformAndQuantiles, MinEntropy, MaxLogSum, Uniform")
        .RequiredArgument("comma separated list of ctr descriptions")
        .Handler1T<TString>([plainJsonPtr](const TString& ctrDescriptionLine) {
            for (const auto& oneCtrConfig : StringSplitter(ctrDescriptionLine).Split(',').SkipEmpty()) {
                (*plainJsonPtr)["simple_ctr"].AppendValue(oneCtrConfig.Token());
            }
            CB_ENSURE(!(*plainJsonPtr)["simple_ctr"].GetArray().empty(), "Empty ctr description " << ctrDescriptionLine);
        });

    parser.AddLongOption("combinations-ctr",
                         "Ctr description should be written in format CtrType[:TargetBorderCount=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num/denum]. CtrType should be one of: Borders, Buckets, BinarizedTargetMeanValue, Counter. TargetBorderType should be one of: Median, GreedyLogSum, UniformAndQuantiles, MinEntropy, MaxLogSum, Uniform")
        .RequiredArgument("comma separated list of ctr descriptions")
        .Handler1T<TString>([plainJsonPtr](const TString& ctrDescriptionLine) {
            for (const auto& oneCtrConfig : StringSplitter(ctrDescriptionLine).Split(',').SkipEmpty()) {
                (*plainJsonPtr)["combinations_ctr"].AppendValue(oneCtrConfig.Token());
            }
            CB_ENSURE(!(*plainJsonPtr)["combinations_ctr"].GetArray().empty(), "Empty ctr description " << ctrDescriptionLine);
        });

    parser.AddLongOption("per-feature-ctr")
        .AddLongName("feature-ctr")
        .RequiredArgument("DESC[;DESC...]")
        .Help("Semicolon separated list of ctr descriptions. Ctr description should be written in format FeatureId:CtrType:[:TargetBorderCount=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num/denum]. CtrType should be one of: Borders, Buckets, BinarizedTargetMeanValue, Counter. TargetBorderType should be one of: Median, GreedyLogSum, UniformAndQuantiles, MinEntropy, MaxLogSum, Uniform")
        .Handler1T<TString>([plainJsonPtr](const TString& ctrDescriptionLine) {
            for (const auto& oneCtrConfig : StringSplitter(ctrDescriptionLine).Split(';').SkipEmpty()) {
                (*plainJsonPtr)["per_feature_ctr"].AppendValue(oneCtrConfig.Token());
            }
            CB_ENSURE(!(*plainJsonPtr)["per_feature_ctr"].GetArray().empty(), "Empty ctr description " << ctrDescriptionLine);
        });

    //legacy fallback
    parser.AddLongOption("ctr",
                         "Ctr description should be written in format FeatureId:CtrType:[:TargetBorderCount=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num/denum]. CtrType should be one of: Borders, Buckets, BinarizedTargetMeanValue, Counter. TargetBorderType should be one of: Median, GreedyLogSum, UniformAndQuantiles, MinEntropy, MaxLogSum, Uniform")
        .RequiredArgument("comma separated list of ctr descriptions")
        .Handler1T<TString>([plainJsonPtr](const TString& ctrDescriptionLine) {
            for (const auto& oneCtrConfig : StringSplitter(ctrDescriptionLine).Split(',').SkipEmpty()) {
                (*plainJsonPtr)["ctr_description"].AppendValue(oneCtrConfig.Token());
            }
            CB_ENSURE(!(*plainJsonPtr)["ctr_description"].GetArray().empty(), "Empty ctr description " << ctrDescriptionLine);
        });

    parser.AddLongOption("ctr-target-border-count", "default border count for target binarization for ctrs")
        .RequiredArgument("int")
        .Handler1T<int>([plainJsonPtr](int count) {
            (*plainJsonPtr)["ctr_target_border_count"] = count;
        });

    const auto counterCalcMethodHelp = TString::Join(
        "Must be one of: ",
        GetEnumAllNames<ECounterCalc>());
    parser.AddLongOption("counter-calc-method", counterCalcMethodHelp)
        .RequiredArgument("method-name")
        .Handler1T<ECounterCalc>([plainJsonPtr](const auto method) {
            (*plainJsonPtr).InsertValue("counter_calc_method", ToString(method));
        });

    parser.AddLongOption("ctr-leaf-count-limit",
                         "Limit maximum ctr leaf count. If there are more leaves than limit, it'll select top values by frequency and put the rest into trashbucket. This option reduces resulting model size and amount of memory used during training. But it might affect the resulting quality. CPU only")
        .RequiredArgument("maxLeafCount")
        .Handler1T<ui64>([plainJsonPtr](ui64 maxLeafCount) {
            (*plainJsonPtr).InsertValue("ctr_leaf_count_limit", maxLeafCount);
        });

    parser.AddLongOption("ctr-history-unit", counterCalcMethodHelp)
        .RequiredArgument("Policy")
        .Handler1T<ECtrHistoryUnit>([plainJsonPtr](const auto unit) {
            (*plainJsonPtr).InsertValue("ctr_history_unit", ToString(unit));
        });

    parser.AddLongOption("store-all-simple-ctr",
                         "Do not limit simple ctr leaves count to topN, store all values from learn set")
        .NoArgument()
        .Handler0([plainJsonPtr]() {
            (*plainJsonPtr).InsertValue("store_all_simple_ctr", true);
        });

    parser.AddLongOption("one-hot-max-size")
        .RequiredArgument("size_t")
        .Handler1T<size_t>([plainJsonPtr](const size_t oneHotMaxSize) {
            (*plainJsonPtr).InsertValue("one_hot_max_size", oneHotMaxSize);
        })
        .Help("Use one-hot encoding for all categorical features with a number of different values less than or equal to the given parameter value. Ctrs are not calculated for such features.");
}

static void ParseDigitizerDescriptions(TStringBuf descriptionLine, TStringBuf idKey, NJson::TJsonValue* digitizers) {
    digitizers->SetType(NJson::EJsonValueType::JSON_ARRAY);
    for (TStringBuf oneConfig : StringSplitter(descriptionLine).Split(',').SkipEmpty()) {
        NJson::TJsonValue digitizer;
        TStringBuf digitizerId, stringParams;
        oneConfig.Split(':', digitizerId, stringParams);
        digitizer[idKey] = digitizerId;
        for (TStringBuf stringParam : StringSplitter(stringParams).Split(':').SkipEmpty()) {
            TStringBuf key, value;
            stringParam.Split('=', key, value);
            digitizer[key] = value;
        }
        digitizers->AppendValue(digitizer);
    }
    CB_ENSURE(!digitizers->GetArray().empty(), "Incorrect description line: " << descriptionLine << ".");
}

static void BindTextFeaturesParams(NLastGetopt::TOpts* parserPtr, NJson::TJsonValue* plainJsonPtr) {
    using namespace NTextProcessing::NDictionary;

    auto& parser = *parserPtr;

    parser.AddLongOption("tokenizers")
        .RequiredArgument("DESC[,DESC...]")
        .Help("Comma separated list of tokenizers descriptions. Description should be written in format "
            "TokenizerId[:optionName=optionValue][:optionName=optionValue]"
        ).Handler1T<TString>([plainJsonPtr](const TString& descriptionLine) {
            ParseDigitizerDescriptions(descriptionLine, "tokenizer_id", &(*plainJsonPtr)["tokenizers"]);
        });

    parser.AddLongOption("dictionaries")
        .RequiredArgument("DESC[,DESC...]")
        .Help("Comma separated list of dictionary descriptions. Description should be written in format "
            "DictionaryId[:occurrence_lower_bound=MinTokenOccurrence][:max_dictionary_size=MaxDictSize][:gram_order=GramOrder][:token_level_type=TokenLevelType]"
        ).Handler1T<TString>([plainJsonPtr](const TString& descriptionLine) {
            ParseDigitizerDescriptions(descriptionLine, "dictionary_id", &(*plainJsonPtr)["dictionaries"]);
        });

    parser.AddLongOption("feature-calcers")
        .RequiredArgument("DESC[,DESC...]")
        .Help("Comma separated list of feature calcers descriptions. Description should be written in format "
            "FeatureCalcerType[:optionName=optionValue][:optionName=optionValue]"
        ).Handler1T<TString>([plainJsonPtr](const TString& descriptionLine) {
            ParseDigitizerDescriptions(descriptionLine, "calcer_type", &(*plainJsonPtr)["feature_calcers"]);
        });

    parser.AddLongOption("text-processing")
        .RequiredArgument("{...}")
        .Help("Text processing json.")
        .Handler1T<TString>([plainJsonPtr](const TString& textProcessingLine) {
            NJson::ReadJsonTree(textProcessingLine, &(*plainJsonPtr)["text_processing"]);
        });
}

static void BindEmbeddingFeaturesParams(NLastGetopt::TOpts* parserPtr, NJson::TJsonValue* plainJsonPtr) {
    auto& parser = *parserPtr;

    parser.AddLongOption("embedding-calcers")
        .RequiredArgument("DESC[,DESC...]")
        .Help("Comma separated list of feature calcers descriptions. Description should be written in format "
              "FeatureCalcerType[:optionName=optionValue][:optionName=optionValue]"
        ).Handler1T<TString>([plainJsonPtr](const TString& descriptionLine) {
            ParseDigitizerDescriptions(descriptionLine, "calcer_type", &(*plainJsonPtr)["embedding_calcers"]);
        });

    parser.AddLongOption("embedding-processing")
        .RequiredArgument("{...}")
        .Help("Embedding processing json.")
        .Handler1T<TString>([plainJsonPtr](const TString& embeddingProcessingLine) {
            NJson::ReadJsonTree(embeddingProcessingLine, &(*plainJsonPtr)["embedding_processing"]);
        });
}

void BindQuantizerDataProcessingParams(NLastGetopt::TOpts* parserPtr, NJson::TJsonValue* plainJsonPtr) {
    auto& parser = *parserPtr;
    parser.AddLongOption('I', "ignore-features",
                         "don't use the specified features in the learn set (the features are separated by colon and can be specified as an inclusive interval, for example: -I 4:78-89:312)")
        .RequiredArgument("INDEXES or NAMES")
        .Handler1T<TString>([plainJsonPtr](const TString& indicesLine) {
            for (const auto& ignoredFeature : StringSplitter(indicesLine).Split(':')) {
                (*plainJsonPtr)["ignored_features"].AppendValue(ignoredFeature.Token());
            }
        });
    parser.AddLongOption("class-names", "names for classes.")
        .RequiredArgument("comma separated list of names")
        .Handler1T<TString>([plainJsonPtr](const TString& namesLine) {
            for (const auto& t : StringSplitter(namesLine).Split(',').SkipEmpty()) {
                (*plainJsonPtr)["class_names"].AppendValue(t.Token());
            }
            CB_ENSURE(!(*plainJsonPtr)["class_names"].GetArray().empty(), "Empty class names list" << namesLine);
        })
        .Help("Takes effect only with classification. Without this parameter classes are 0, 1, ..., classes-count - 1");
}

void BindDataProcessingParams(NLastGetopt::TOpts* parserPtr, NJson::TJsonValue* plainJsonPtr) {
    BindQuantizerDataProcessingParams(parserPtr, plainJsonPtr);
    auto& parser = *parserPtr;
    parser.AddLongOption("has-time", "Use dataset order as time")
        .NoArgument()
        .Handler0([plainJsonPtr]() {
            (*plainJsonPtr)["has_time"] = true;
        });

    parser.AddLongOption("allow-const-label", "Allow const label")
        .NoArgument()
        .Handler0([plainJsonPtr]() {
            (*plainJsonPtr)["allow_const_label"] = true;
        });

    parser.AddLongOption("target-border", "Border for target binarization")
        .RequiredArgument("float")
        .Handler1T<float>([plainJsonPtr](float targetBorder) {
            (*plainJsonPtr)["target_border"] = targetBorder;
        });

    parser.AddLongOption("classes-count", "number of classes")
        .RequiredArgument("int")
        .Handler1T<int>([plainJsonPtr](const int classesCount) {
            (*plainJsonPtr).InsertValue("classes_count", classesCount);
        })
        .Help("Takes effect only with multiclassification. If classes-count is given (and class-names is not given), then each class label should be less than that number.");

    parser.AddLongOption("class-weights", "Weights for classes.")
        .RequiredArgument("comma separated list of weights")
        .Handler1T<TString>([plainJsonPtr](const TString& weightsLine) {
            for (const auto& t : StringSplitter(weightsLine).Split(',').SkipEmpty()) {
                (*plainJsonPtr)["class_weights"].AppendValue(FromString<float>(t.Token()));
            }
            CB_ENSURE(!(*plainJsonPtr)["class_weights"].GetArray().empty(), "Empty class weights list " << weightsLine);
        })
        .Help("Takes effect only with classification. Number of classes indicated by classes-count, class-names and class-weights should be the same");

    const auto autoClassWeightsHelp = TString::Join(
        "Takes effect only with classification. Must be one of: ",
        GetEnumAllNames<EAutoClassWeightsType>(),
        ". Default: ",
        ToString(EAutoClassWeightsType::None));

    parser.AddLongOption("auto-class-weights")
        .RequiredArgument("String")
        .Handler1T<EAutoClassWeightsType>([plainJsonPtr](const auto classWeightsType){
            (*plainJsonPtr)["auto_class_weights"] = ToString(classWeightsType);
        })
        .Help(autoClassWeightsHelp);

    parser.AddLongOption("force-unit-auto-pair-weights", "Set weight to 1 for all auto-generated pairs rather than use group weight")
        .NoArgument()
        .Handler0([plainJsonPtr]() {
            (*plainJsonPtr)["force_unit_auto_pair_weights"] = true;
        });

    const auto gpuCatFeatureStorageHelp = TString::Join(
        "GPU only. Must be one of: ",
        GetEnumAllNames<EGpuCatFeaturesStorage>(),
        ". Default: ",
        ToString(EGpuCatFeaturesStorage::GpuRam));
    parser
        .AddLongOption("gpu-cat-features-storage", gpuCatFeatureStorageHelp)
        .RequiredArgument("String")
        .Handler1T<EGpuCatFeaturesStorage>([plainJsonPtr](const auto storage) {
            (*plainJsonPtr)["gpu_cat_features_storage"] = ToString(storage);
        });

    parser.AddLongOption("dev-leafwise-scoring", "Use scoring with sorting by leaf")
        .NoArgument()
        .Handler0([plainJsonPtr](){
            (*plainJsonPtr)["dev_leafwise_scoring"] = true;
        });

    parser
        .AddLongOption("dev-group-features")
        .NoArgument()
        .Handler0([plainJsonPtr]() {
            (*plainJsonPtr)["dev_group_features"] = true;
        });
}

static void BindDistributedTrainingParams(NLastGetopt::TOpts* parserPtr, NJson::TJsonValue* plainJsonPtr) {
    auto& parser = *parserPtr;
    const auto nodeTypeHelp = TString::Join("Must be one of: ", GetEnumAllNames<ENodeType>());
    parser
        .AddLongOption("node-type", nodeTypeHelp)
        .RequiredArgument("String")
        .Handler1T<ENodeType>([plainJsonPtr](const auto nodeType) {
            (*plainJsonPtr)["node_type"] = ToString(nodeType);
        });

    parser
        .AddLongOption("node-port")
        .RequiredArgument("int")
        .Help("TCP port for this worker; default is 0")
        .Handler1T<int>([plainJsonPtr](int nodePort) {
            (*plainJsonPtr)["node_port"] = nodePort;
        });

    parser
        .AddLongOption("file-with-hosts")
        .RequiredArgument("String")
        .Help("File listing <worker's IP address>:<worker's TCP port> for all workers employed by this master; default is hosts.txt")
        .Handler1T<TString>([plainJsonPtr](const TString& nodeFile) {
            (*plainJsonPtr)["file_with_hosts"] = nodeFile;
        });
}

static void BindSystemParams(NLastGetopt::TOpts* parserPtr, NJson::TJsonValue* plainJsonPtr) {
    auto& parser = *parserPtr;
    parser.AddCharOption('T', "worker thread count (default: core count)")
        .AddLongName("thread-count")
        .RequiredArgument("count")
        .Handler1T<int>([plainJsonPtr](int count) {
            (*plainJsonPtr).InsertValue("thread_count", count);
        });

    parser.AddLongOption("used-ram-limit", "Try to limit used memory. CPU only. WARNING: This option affects CTR memory usage only.\nAllowed suffixes: GB, MB, KB in different cases")
            .RequiredArgument("TARGET_RSS")
            .Handler1T<TString>([plainJsonPtr](const TString& param) {
                (*plainJsonPtr)["used_ram_limit"] = param;
            });

    parser
            .AddLongOption("gpu-ram-part")
            .RequiredArgument("double")
            .Help("Fraction of GPU memory to use. Should be in range (0, 1]")
            .Handler1T<double>([plainJsonPtr](const double part) {
                (*plainJsonPtr)["gpu_ram_part"] = part;
            });

    parser
        .AddLongOption("devices")
        .RequiredArgument("String")
        .Help("List of devices. Could be enumeration with : separator (1:2:4), range 1-3; 1-3:5. Default -1 (use all devices)")
        .Handler1T<TString>([plainJsonPtr](const TString& devices) {
            (*plainJsonPtr)["devices"] = devices;
        });

    parser
            .AddLongOption("pinned-memory-size")
            .RequiredArgument("String")
            .Help("GPU only. Minimum CPU pinned memory to use, e.g. 8gb, 100000, etc. Valid suffixes are tb, gb, mb, kb, b")
            .Handler1T<TString>([plainJsonPtr](const TString& param) {
                (*plainJsonPtr)["pinned_memory_size"] = param;
            });
}

void BindQuantizerBinarizationParams(NLastGetopt::TOpts* parserPtr, NJson::TJsonValue* plainJsonPtr) {
    auto& parser = *parserPtr;
    parser.AddLongOption('x', "border-count", "count of borders per float feature. Should be in range [1, 255]")
        .RequiredArgument("int")
        .Handler1T<int>([plainJsonPtr](int count) {
            (*plainJsonPtr)["border_count"] = count;
        });
    parser.AddLongOption("per-float-feature-quantization")
      .AddLongName("per-float-feature-binarization") // TODO(kirillovs): remove alias when all users switch to new one
      .RequiredArgument("DESC[;DESC...]")
      .Help("Semicolon separated list of float binarization descriptions. Float binarization description should be written in format FeatureId[:border_count=BorderCount][:nan_mode=BorderType][:border_type=border_selection_method]")
      .Handler1T<TString>([plainJsonPtr](const TString& ctrDescriptionLine) {
          for (const auto& oneCtrConfig : StringSplitter(ctrDescriptionLine).Split(';').SkipEmpty()) {
              (*plainJsonPtr)["per_float_feature_quantization"].AppendValue(oneCtrConfig.Token());
          }
          CB_ENSURE(!(*plainJsonPtr)["per_float_feature_quantization"].GetArray().empty(), "Empty per float feature quantization settings " << ctrDescriptionLine);
      });

    const auto featureBorderTypeHelp = TString::Join(
        "Must be one of: ",
        GetEnumAllNames<EBorderSelectionType>());
    parser.AddLongOption('g', "feature-border-type", featureBorderTypeHelp)
        .AddLongName("grid")
        .RequiredArgument("border-type")
        .Handler1T<EBorderSelectionType>([plainJsonPtr](const auto type) {
            (*plainJsonPtr)["feature_border_type"] = ToString(type);
        });

    const auto nanModeHelp = TString::Join(
        "Must be one of: ",
        GetEnumAllNames<ENanMode>(),
        " Default: ",
        ToString(ENanMode::Min));
    parser.AddLongOption("nan-mode", nanModeHelp)
        .RequiredArgument("nan-mode")
        .Handler1T<ENanMode>([plainJsonPtr](const auto nanMode) {
            (*plainJsonPtr)["nan_mode"] = ToString(nanMode);
        });
}

static void BindBinarizationParams(NLastGetopt::TOpts* parserPtr, NJson::TJsonValue* plainJsonPtr) {
    BindQuantizerBinarizationParams(parserPtr, plainJsonPtr);
    auto& parser = *parserPtr;
    parser.AddLongOption("dev-max-subset-size-for-build-borders", "Maximum size of subset for build borders algorithm. Default: 200000")
        .RequiredArgument("int")
        .Handler1T<int>([plainJsonPtr](const int maxSubsetSizeForBuildBorders) {
          (*plainJsonPtr)["dev_max_subset_size_for_build_borders"] = maxSubsetSizeForBuildBorders;
        });
}

static void BindCatboostParams(NLastGetopt::TOpts* parserPtr, NJson::TJsonValue* plainJsonPtr) {
    auto& parser = *parserPtr;
    parser.AddLongOption('r', "seed")
        .AddLongName("random-seed")
        .RequiredArgument("count")
        .Handler1T<ui64>([plainJsonPtr](ui64 seed) {
            (*plainJsonPtr)["random_seed"] = seed;
        });

    const auto taskTypeHelp = TString::Join("Must be one of: ", GetEnumAllNames<ETaskType>());
    parser
        .AddLongOption("task-type", taskTypeHelp)
        .RequiredArgument("String")
        .Handler1T<ETaskType>([plainJsonPtr](const auto taskType) {
            (*plainJsonPtr)["task_type"] = ToString(taskType);
        });

    parser
        .AddLongOption("logging-level")
        .RequiredArgument("Level")
        .Help("Logging level: one of (Silent, Verbose, Info, Debug)")
        .Handler1T<TString>([plainJsonPtr](const TString& level) {
            (*plainJsonPtr)["logging_level"] = level;
        });

    parser.AddLongOption("detailed-profile", "use detailed profile")
        .NoArgument()
        .Handler0([plainJsonPtr]() {
            (*plainJsonPtr)["detailed_profile"] = true;
        });
}

static void ParseMetadata(int argc, const char* argv[], NLastGetopt::TOpts* parserPtr, NJson::TJsonValue* plainJsonPtr) {
    auto& parser = *parserPtr;
    bool setModelMetadata = false;
    parser.AddLongOption("set-metadata-from-freeargs", "treat [key value] freeargs pairs as model metadata")
        .StoreValue(&setModelMetadata, true)
        .NoArgument();
    if (argc == 1) {
        parser.PrintUsage(GetExecPath(), Cerr);
        std::exit(-1);
    }
    NLastGetopt::TOptsParseResult parserResult{&parser, argc, argv};
    if (!setModelMetadata) {
        CB_ENSURE(
            parserResult.GetFreeArgCount() == 0,
            "freearg '" << parserResult.GetFreeArgs()[0] << "' is misplaced, "
            "or a long option name is preceeded with single -; "
            "to use freeargs, put --set-metadata-from-freeargs before the 1st freearg.");
    } else {
        auto freeArgs = parserResult.GetFreeArgs();
        auto freeArgCount = freeArgs.size();
        auto& metadata = (*plainJsonPtr)["metadata"];
        CB_ENSURE(freeArgCount % 2 == 0, "key-value freeargs count should be even");
        for (size_t i = 0; i < freeArgCount; i += 2) {
            metadata[freeArgs[i]] = freeArgs[i + 1];
        }
    }
}

void ParseCommandLine(int argc, const char* argv[],
                      NJson::TJsonValue* plainJsonPtr,
                      TString* paramsPath,
                      NCatboostOptions::TPoolLoadParams* params) {
    auto parser = NLastGetopt::TOpts();
    parser.ArgPermutation_ = NLastGetopt::EArgPermutation::REQUIRE_ORDER;
    parser.AddHelpOption();
    BindPoolLoadParams(&parser, params);

    parser
        .AddLongOption("trigger-core-dump")
        .NoArgument()
        .Handler0([] { CB_ENSURE(false, "Aborting on user request"); })
        .Help("Trigger core dump")
        .Hidden();

    parser.AddLongOption("params-file", "Path to JSON file with params.")
        .RequiredArgument("PATH")
        .StoreResult(paramsPath)
        .Help("If param is given in json file and in command line then one from command line will be used.");

    BindMetricParams(&parser, plainJsonPtr);

    BindOutputParams(&parser, plainJsonPtr);

    BindBoostingParams(&parser, plainJsonPtr);

    BindTreeParams(&parser, plainJsonPtr);

    BindCatFeatureParams(&parser, plainJsonPtr);

    BindTextFeaturesParams(&parser, plainJsonPtr);

    BindEmbeddingFeaturesParams(&parser, plainJsonPtr);

    BindDataProcessingParams(&parser, plainJsonPtr);

    BindBinarizationParams(&parser, plainJsonPtr);

    BindSystemParams(&parser, plainJsonPtr);

    BindDistributedTrainingParams(&parser, plainJsonPtr);

    BindCatboostParams(&parser, plainJsonPtr);

    ParseMetadata(argc, argv, &parser, plainJsonPtr);
}

void ParseModelBasedEvalCommandLine(
    int argc,
    const char* argv[],
    NJson::TJsonValue* plainJsonPtr,
    TString* paramsPath,
    NCatboostOptions::TPoolLoadParams* params
) {
    auto parser = NLastGetopt::TOpts();
    parser.ArgPermutation_ = NLastGetopt::EArgPermutation::REQUIRE_ORDER;
    parser.AddHelpOption();
    BindPoolLoadParams(&parser, params);

    parser.AddLongOption("params-file", "Path to JSON file with params.")
        .RequiredArgument("PATH")
        .StoreResult(paramsPath)
        .Help("If param is given in json file and in command line then one from command line will be used.");

    BindMetricParams(&parser, plainJsonPtr);

    BindOutputParams(&parser, plainJsonPtr);

    BindBoostingParams(&parser, plainJsonPtr);

    BindModelBasedEvalParams(&parser, plainJsonPtr);

    BindTreeParams(&parser, plainJsonPtr);

    BindCatFeatureParams(&parser, plainJsonPtr);

    BindTextFeaturesParams(&parser, plainJsonPtr);

    BindEmbeddingFeaturesParams(&parser, plainJsonPtr);

    BindDataProcessingParams(&parser, plainJsonPtr);

    BindBinarizationParams(&parser, plainJsonPtr);

    BindSystemParams(&parser, plainJsonPtr);

    BindCatboostParams(&parser, plainJsonPtr);

    NLastGetopt::TOptsParseResult parserResult{&parser, argc, argv};
}

void ParseFeatureEvalCommandLine(
    int argc,
    const char* argv[],
    NJson::TJsonValue* plainJsonPtr,
    NJson::TJsonValue* featureEvalOptions,
    TString* paramsPath,
    NCatboostOptions::TPoolLoadParams* params
) {
    auto parser = NLastGetopt::TOpts();
    parser.ArgPermutation_ = NLastGetopt::EArgPermutation::REQUIRE_ORDER;
    parser.AddHelpOption();
    BindPoolLoadParams(&parser, params);

    parser.AddLongOption("params-file", "Path to JSON file with params.")
        .RequiredArgument("PATH")
        .StoreResult(paramsPath)
        .Help("If param is given in json file and in command line then one from command line will be used.");

    BindMetricParams(&parser, plainJsonPtr);

    BindOutputParams(&parser, plainJsonPtr);

    BindBoostingParams(&parser, plainJsonPtr);

    BindFeatureEvalParams(&parser, featureEvalOptions);

    BindTreeParams(&parser, plainJsonPtr);

    BindCatFeatureParams(&parser, plainJsonPtr);

    BindTextFeaturesParams(&parser, plainJsonPtr);

    BindEmbeddingFeaturesParams(&parser, plainJsonPtr);

    BindDataProcessingParams(&parser, plainJsonPtr);

    BindBinarizationParams(&parser, plainJsonPtr);

    BindSystemParams(&parser, plainJsonPtr);

    BindCatboostParams(&parser, plainJsonPtr);

    ParseMetadata(argc, argv, &parser, plainJsonPtr);
}


void ParseFeaturesSelectCommandLine(
    int argc,
    const char* argv[],
    NJson::TJsonValue* plainJsonPtr,
    TString* paramsPath,
    NCatboostOptions::TPoolLoadParams* params
) {
    auto parser = NLastGetopt::TOpts();
    parser.ArgPermutation_ = NLastGetopt::EArgPermutation::REQUIRE_ORDER;
    parser.AddHelpOption();
    BindPoolLoadParams(&parser, params);

    parser.AddLongOption("params-file", "Path to JSON file with params.")
        .RequiredArgument("PATH")
        .StoreResult(paramsPath)
        .Help("If param is given in json file and in command line then one from command line will be used.");

    BindMetricParams(&parser, plainJsonPtr);

    BindOutputParams(&parser, plainJsonPtr);

    BindBoostingParams(&parser, plainJsonPtr);

    BindFeaturesSelectParams(&parser, plainJsonPtr);

    BindTreeParams(&parser, plainJsonPtr);

    BindCatFeatureParams(&parser, plainJsonPtr);

    BindTextFeaturesParams(&parser, plainJsonPtr);

    BindEmbeddingFeaturesParams(&parser, plainJsonPtr);

    BindDataProcessingParams(&parser, plainJsonPtr);

    BindBinarizationParams(&parser, plainJsonPtr);

    BindSystemParams(&parser, plainJsonPtr);

    BindDistributedTrainingParams(&parser, plainJsonPtr);

    BindCatboostParams(&parser, plainJsonPtr);

    ParseMetadata(argc, argv, &parser, plainJsonPtr);
}
