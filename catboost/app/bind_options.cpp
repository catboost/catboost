#include "bind_options.h"

#include <catboost/libs/column_description/column.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/libs/options/analytical_mode_params.h>
#include <catboost/libs/options/catboost_options.h>
#include <catboost/libs/options/enums.h>
#include <catboost/libs/options/enum_helpers.h>
#include <catboost/libs/options/output_file_options.h>
#include <catboost/libs/options/plain_options_helper.h>

#include <library/getopt/small/last_getopt_opts.h>
#include <library/grid_creator/binarization.h>
#include <library/json/json_reader.h>
#include <library/logger/log.h>

#include <util/generic/algorithm.h>
#include <util/generic/serialized_enum.h>
#include <util/generic/strbuf.h>
#include <util/generic/vector.h>
#include <util/string/cast.h>
#include <util/string/iterator.h>
#include <util/string/join.h>
#include <util/string/split.h>
#include <util/stream/file.h>
#include <util/system/fs.h>
#include <util/system/yassert.h>


using namespace NCB;

void InitOptions(
    const TString& optionsFile,
    NJson::TJsonValue* catBoostJsonOptions,
    NJson::TJsonValue* outputOptionsJson
) {
    if (!optionsFile.empty()) {
        CB_ENSURE(NFs::Exists(optionsFile), "Params file does not exist " << optionsFile);
        TIFStream in(optionsFile);
        NJson::TJsonValue fromOptionsFile;
        CB_ENSURE(NJson::ReadJsonTree(&in, &fromOptionsFile), "can't parse params file");
        NCatboostOptions::PlainJsonToOptions(fromOptionsFile, catBoostJsonOptions, outputOptionsJson);
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

inline static TVector<int> ParseIndicesLine(const TStringBuf indicesLine) {
    TVector<int> result;
    for (const auto& t : StringSplitter(indicesLine).Split(':')) {
        const auto s = t.Token();
        int from = FromString<int>(s.Before('-'));
        int to = FromString<int>(s.After('-')) + 1;
        for (int i = from; i < to; ++i) {
            result.push_back(i);
        }
    }
    return result;
}

inline static void BindPoolLoadParams(NLastGetopt::TOpts* parser, NCatboostOptions::TPoolLoadParams* loadParamsPtr) {
    BindDsvPoolFormatParams(parser, &(loadParamsPtr->DsvPoolFormatParams));

    parser->AddLongOption('f', "learn-set", "learn set path")
        .RequiredArgument("[SCHEME://]PATH")
        .Handler1T<TStringBuf>([loadParamsPtr](const TStringBuf& str) {
            loadParamsPtr->LearnSetPath = TPathWithScheme(str, "dsv");
        });

    parser->AddLongOption('t', "test-set", "path to one or more test sets")
        .RequiredArgument("[SCHEME://]PATH[,[SCHEME://]PATH...]")
        .Handler1T<TStringBuf>([loadParamsPtr](const TStringBuf& str) {
            for (const auto& path : StringSplitter(str).Split(',').SkipEmpty()) {
                if (!path.Empty()) {
                    loadParamsPtr->TestSetPaths.emplace_back(path.Token().ToString(), "dsv");
                }
            }
            CB_ENSURE(!loadParamsPtr->TestSetPaths.empty(), "Empty test path");
        });

    parser->AddLongOption("learn-pairs", "path to learn pairs")
        .RequiredArgument("[SCHEME://]PATH")
        .Handler1T<TStringBuf>([loadParamsPtr](const TStringBuf& str) {
            loadParamsPtr->PairsFilePath = TPathWithScheme(str, "file");
        });

    parser->AddLongOption("test-pairs", "path to test pairs")
        .RequiredArgument("[SCHEME://]PATH")
        .Handler1T<TStringBuf>([loadParamsPtr](const TStringBuf& str) {
            loadParamsPtr->TestPairsFilePath = TPathWithScheme(str, "file");
        });

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
            if (cvType == ECrossValidation::Classical || cvType == ECrossValidation::Inverted) {
                Split(params, ';', loadParamsPtr->CvParams.FoldIdx, loadParamsPtr->CvParams.FoldCount);
                loadParamsPtr->CvParams.Inverted = (cvType == ECrossValidation::Inverted);
            }
            loadParamsPtr->CvParams.Check();
        });

    parser->AddLongOption("cv-rand", "cross-validation random seed")
        .RequiredArgument("seed")
        .StoreResult(&loadParamsPtr->CvParams.PartitionRandSeed);

    parser->AddLongOption("input-borders-file", "file with borders")
            .RequiredArgument("PATH")
            .StoreResult(&loadParamsPtr->BordersFile);
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
            const auto enum_ = FromString<ELossFunction>(TStringBuf(value).Before(':'));
            CB_ENSURE(IsIn(allObjectives, enum_), "objective is not allowed");
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
            "Should be one of: ", GetEnumAllNames<EFstrType >());
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


    parser.AddLongOption("growing-policy", "Tree growing policy")
            .RequiredArgument("Type (ObliviousTree, Region,…)")
            .Handler1T<TString>([plainJsonPtr](const TString& policy) {
                (*plainJsonPtr)["growing_policy"] = policy;
            });

    parser.AddLongOption("max-leaves-count", "Max leaves count")
        .RequiredArgument("INT")
        .Handler1T<ui32>([plainJsonPtr](const ui32 maxLeavesCount) {
            (*plainJsonPtr)["max_leaves_count"] = maxLeavesCount;
        });

    parser.AddLongOption("min-samples-in-leaf", "Minimum number of samples in leaf")
        .RequiredArgument("Double")
        .Handler1T<double>([plainJsonPtr](double minSamples) {
            (*plainJsonPtr)["min_samples_in_leaf"] = minSamples;
        });

    parser.AddLongOption("l2-leaf-reg", "Regularization value. Should be >= 0")
        .RequiredArgument("float")
        .Handler1T<float>([plainJsonPtr](float reg) {
            (*plainJsonPtr)["l2_leaf_reg"] = reg;
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

    parser.AddLongOption("random-strength")
        .RequiredArgument("float")
        .Handler1T<float>([plainJsonPtr](float randomStrength) {
            (*plainJsonPtr)["random_strength"] = randomStrength;
        })
        .Help("score stdandart deviation multiplier");

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
        ". By default CatBoost uses bayesian bootstrap type");
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
        .Help("Controls sample rate for bagging. Could be used iff bootstrap-type is Poisson, Bernoulli. Possible values are from (0, 1]; 0.66 by default."
        );

    parser
        .AddLongOption("observations-to-bootstrap")
        .RequiredArgument("FLAG")
        .Help("GPU only.Use bootstraped weights on learn and test folds. By default bootstrap used only for test fold part.")
        .Handler1T<TString>([plainJsonPtr](const TString& type) {
            (*plainJsonPtr)["observations_to_bootstrap"] = type;
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
        .Help("If parameter is specified than features with no more than specified value different values will be converted to float features using one-hot encoding. No ctrs will be calculated on this features.");
}

static void BindDataProcessingParams(NLastGetopt::TOpts* parserPtr, NJson::TJsonValue* plainJsonPtr) {
    auto& parser = *parserPtr;
    parser.AddLongOption('I', "ignore-features",
                         "don't use the specified features in the learn set (the features are separated by colon and can be specified as an inclusive interval, for example: -I 4:78-89:312)")
        .RequiredArgument("INDEXES")
        .Handler1T<TString>([plainJsonPtr](const TString& indicesLine) {
            auto ignoredFeatures = ParseIndicesLine(indicesLine);
            for (int f : ignoredFeatures) {
                (*plainJsonPtr)["ignored_features"].AppendValue(f);
            }
        });

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

    parser.AddLongOption("classes-count", "number of classes")
        .RequiredArgument("int")
        .Handler1T<int>([plainJsonPtr](const int classesCount) {
            (*plainJsonPtr).InsertValue("classes_count", classesCount);
        })
        .Help("Takes effect only with MultiClass loss function. If classes-count is given (and class-names is not given), then each class label should be less than that number.");

    parser.AddLongOption("class-names", "names for classes.")
        .RequiredArgument("comma separated list of names")
        .Handler1T<TString>([plainJsonPtr](const TString& namesLine) {
            for (const auto& t : StringSplitter(namesLine).Split(',').SkipEmpty()) {
                (*plainJsonPtr)["class_names"].AppendValue(t.Token());
            }
            CB_ENSURE(!(*plainJsonPtr)["class_names"].GetArray().empty(), "Empty class names list" << namesLine);
        })
        .Help("Takes effect only with MultiClass/LogLoss loss functions. Wihout this parameter classes are 0, 1, ..., classes-count - 1");

    parser.AddLongOption("class-weights", "Weights for classes.")
        .RequiredArgument("comma separated list of weights")
        .Handler1T<TString>([plainJsonPtr](const TString& weightsLine) {
            for (const auto& t : StringSplitter(weightsLine).Split(',').SkipEmpty()) {
                (*plainJsonPtr)["class_weights"].AppendValue(FromString<float>(t.Token()));
            }
            CB_ENSURE(!(*plainJsonPtr)["class_weights"].GetArray().empty(), "Empty class weights list " << weightsLine);
        })
        .Help("Takes effect only with MultiClass/LogLoss loss functions. Number of classes indicated by classes-count, class-names and class-weights should be the same");

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
            .RequiredArgument("int")
            .Help("GPU only. Minimum CPU pinned memory to use")
            .Handler1T<TString>([plainJsonPtr](const TString& param) {
                (*plainJsonPtr)["pinned_memory_size"] = param;
            });
}

static void BindBinarizationParams(NLastGetopt::TOpts* parserPtr, NJson::TJsonValue* plainJsonPtr) {
    auto& parser = *parserPtr;
    parser.AddLongOption('x', "border-count", "count of borders per float feature. Should be in range [1, 255]")
        .RequiredArgument("int")
        .Handler1T<int>([plainJsonPtr](int count) {
            (*plainJsonPtr)["border_count"] = count;
        });

    const auto featureBorderTypeHelp = TString::Join(
        "Must be one of: ",
        GetEnumAllNames<EBorderSelectionType>());
    parser.AddLongOption("feature-border-type", featureBorderTypeHelp)
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

void ParseCommandLine(int argc, const char* argv[],
                      NJson::TJsonValue* plainJsonPtr,
                      TString* paramsPath,
                      NCatboostOptions::TPoolLoadParams* params) {
    auto parser = NLastGetopt::TOpts();
    parser.AddHelpOption();
    BindPoolLoadParams(&parser, params);

    parser
        .AddLongOption("trigger-core-dump")
        .NoArgument()
        .Handler0([] { Y_FAIL("Aborting on user request"); })
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

    BindDataProcessingParams(&parser, plainJsonPtr);

    BindBinarizationParams(&parser, plainJsonPtr);

    BindSystemParams(&parser, plainJsonPtr);

    BindDistributedTrainingParams(&parser, plainJsonPtr);

    BindCatboostParams(&parser, plainJsonPtr);

    bool setModelMetadata = false;
    parser.AddLongOption("set-metadata-from-freeargs", "treat [key value] freeargs pairs as model metadata")
        .StoreValue(&setModelMetadata, true)
        .NoArgument();

    NLastGetopt::TOptsParseResult parserResult{&parser, argc, argv};
    if (!setModelMetadata) {
        CB_ENSURE(parserResult.GetFreeArgCount() == 0, "use \"--set-metadata-from-freeargs\" to enable freeargs");
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
