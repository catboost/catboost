#include "bind_options.h"

#include <catboost/libs/column_description/column.h>

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


inline static ui64 ParseMemorySizeDescription(const TString& memSizeDescription) {
    TString sizeLine = memSizeDescription;
    ui64 sizeMultiplier = 1;
    if (sizeLine.back() == 'b' || sizeLine.back() == 'B') {
        sizeLine.pop_back();
        switch (sizeLine.back()) {
            case 'k':
            case 'K':
                sizeMultiplier = 1024;
                break;
            case 'm':
            case 'M':
                sizeMultiplier = 1024 * 1024;
                break;
            case 'g':
            case 'G':
                sizeMultiplier = 1024 * 1024 * 1024;
                break;
            default:
                CB_ENSURE(false, "unknown size suffix: " << memSizeDescription);
        }
        sizeLine.pop_back();
    }
    return sizeMultiplier * FromString<ui64>(sizeLine);
}

inline static void BindPoolLoadParams(NLastGetopt::TOpts* parser, NCatboostOptions::TPoolLoadParams* loadParamsPtr) {
    parser->AddLongOption('f', "learn-set", "learn set path")
        .RequiredArgument("PATH")
        .StoreResult(&loadParamsPtr->LearnFile);

    parser->AddLongOption('t', "test-set", "test set path")
        .RequiredArgument("PATH")
        .DefaultValue("")
        .StoreResult(&loadParamsPtr->TestFile);

    parser->AddLongOption("learn-pairs", "path to learn pairs")
        .RequiredArgument("PATH")
        .DefaultValue("")
        .StoreResult(&loadParamsPtr->PairsFile);

    parser->AddLongOption("test-pairs", "path to test pairs")
        .RequiredArgument("PATH")
        .DefaultValue("")
        .StoreResult(&loadParamsPtr->TestPairsFile);

    parser->AddLongOption("column-description", "column desctiption file name")
        .AddLongName("cd")
        .RequiredArgument("PATH")
        .StoreResult(&loadParamsPtr->CdFile);

    parser->AddLongOption("delimiter", "Learning and training sets delimiter")
        .RequiredArgument("SYMBOL")
        .Handler1T<TString>([loadParamsPtr](const TString& oneChar) {
            CB_ENSURE(oneChar.size() == 1, "only single char delimiters supported");
            loadParamsPtr->Delimiter = oneChar[0];
        });

    parser->AddLongOption("has-header", "Read first line as header")
        .NoArgument()
        .StoreValue(&loadParamsPtr->HasHeader,
                    true);

    parser->AddCharOption('X', "cross validation, test on fold n of k, n is 0-based")
        .RequiredArgument("n/k")
        .Handler1T<TStringBuf>([loadParamsPtr](const TStringBuf& str) {
            Split(str,
                  '/', loadParamsPtr->CvParams.FoldIdx, loadParamsPtr->CvParams.FoldCount);
            loadParamsPtr->CvParams.Inverted = false;
        });

    parser->AddCharOption('Y', "inverted cross validation, train on fold n of k, n is 0-based")
        .RequiredArgument("n/k")
        .Handler1T<TStringBuf>([loadParamsPtr](const TStringBuf& str) {
            Split(str,
                  '/', loadParamsPtr->CvParams.FoldIdx, loadParamsPtr->CvParams.FoldCount);
            loadParamsPtr->CvParams.Inverted = true;
        });

    parser->AddLongOption("cv-rand", "cross-validation random seed")
        .RequiredArgument("seed")
        .StoreResult(&loadParamsPtr
                          ->CvParams.RandSeed);
}

void ParseCommandLine(int argc, const char* argv[],
                      NJson::TJsonValue* plainJsonPtr,
                      TString* paramsPath,
                      NCatboostOptions::TPoolLoadParams* params) {
    auto parser = NLastGetopt::TOpts();
    parser.AddHelpOption();
    BindPoolLoadParams(&parser, params);

    parser.AddLongOption("loss-function",
                         "Should be one of: Logloss, CrossEntropy, RMSE, MAE, Quantile, LogLinQuantile, MAPE, Poisson, MultiClass, MultiClassOneVsAll, PairLogit, QueryRMSE, QuerySoftMax. A loss might have params, then params should be written in format Loss:paramName=value.")
        .RequiredArgument("string")
        .Handler1T<TString>([plainJsonPtr](const TString& lossDescription) {
            (*plainJsonPtr)["loss_function"] = lossDescription;
        });

    parser.AddLongOption("custom-metric",
                         "A metric might have params, then params should be written in format Loss:paramName=value. Loss should be one of: Logloss, CrossEntropy, RMSE, MAE, Quantile, LogLinQuantile, MAPE, Poisson, MultiClass, MultiClassOneVsAll, PairLogit, QueryRMSE, QuerySoftMax, R2, AUC, Accuracy, Precision, Recall, F1, TotalF1, MCC, PairAccuracy")
            .AddLongName("custom-loss")
        .RequiredArgument("comma separated list of metric functions")
        .Handler1T<TString>([plainJsonPtr](const TString& lossFunctionsLine) {
            for (const auto& lossFunction : StringSplitter(lossFunctionsLine).Split(',')) {
                (*plainJsonPtr)["custom_metric"].AppendValue(NJson::TJsonValue(lossFunction.Token()));
            }
        });

    parser.AddLongOption("eval-metric")
        .RequiredArgument("string")
        .Handler1T<TString>([plainJsonPtr](const TString& metric) {
            (*plainJsonPtr)["eval_metric"] = metric;
        })
        .Help("evaluation metric for overfitting detector (if enabled) and best model "
              "selection in format MetricName:param=value. If not specified default metric for objective is used.");

    //output files
    parser.AddLongOption('m', "model-file", "model file name")
        .RequiredArgument("PATH")
        .Handler1T<TString>([plainJsonPtr](const TString& name) {
            (*plainJsonPtr)["result_model_file"] = name;
        });

    parser.AddLongOption("eval-file", "eval output file name")
        .RequiredArgument("PATH")
        .Handler1T<TString>([plainJsonPtr](const TString& name) {
            (*plainJsonPtr)["eval_file_name"] = name;
        });

    parser.AddLongOption("fstr-file", "Save fstr to this file")
        .RequiredArgument("filename")
        .Handler1T<TString>([plainJsonPtr](const TString& name) {
            (*plainJsonPtr)["fstr_regular_file"] = name;
        });

    parser.AddLongOption("fstr-internal-file", "Save internal fstr values to this file")
        .RequiredArgument("filename")
        .Handler1T<TString>([plainJsonPtr](const TString& name) {
            (*plainJsonPtr)["fstr_internal_file"] = name;
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

    parser.AddLongOption("use-best-model", "If true - save all trees until best iteration on test.")
        .RequiredArgument("bool")
        .Handler1T<TString>([plainJsonPtr](const TString& useBestModel) {
            (*plainJsonPtr)["use_best_model"] = FromString<bool>(useBestModel);
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

    parser.AddLongOption("metric-period", "period of printing metrics to stdout")
        .RequiredArgument("int")
        .Handler1T<TString>([plainJsonPtr](const TString& period) {
        (*plainJsonPtr)["metric_period"] = FromString<int>(period);
    });

    parser.AddLongOption("snapshot-file", "use progress file for restoring progress after crashes")
        .RequiredArgument("PATH")
        .Handler1T<TString>([plainJsonPtr](const TString& path) {
            (*plainJsonPtr)["save_snapshot"] = true;
            (*plainJsonPtr)["snapshot_file"] = path;
        });

    parser.AddLongOption("output-columns")
            .RequiredArgument("Comma separated list of column indexes")
            .Handler1T<TString>([plainJsonPtr](const TString& indexesLine) {
                (*plainJsonPtr)["output_columns"] = NULL;
                for (const auto& t : StringSplitter(indexesLine).Split(',')) {
                    (*plainJsonPtr)["output_columns"].AppendValue(t.Token());

                }
            });

    parser.AddLongOption("prediction-type")
        .RequiredArgument("Comma separated list of prediction types. Every prediction type should be one of: Probability, Class, RawFormulaVal. CPU only")
        .Handler1T<TString>([plainJsonPtr](const TString& predictionTypes) {
            (*plainJsonPtr)["output_columns"].AppendValue("DocId");
            for (const auto& t : StringSplitter(predictionTypes).Split(',')) {
                (*plainJsonPtr)["prediction_type"].AppendValue(t.Token());
                (*plainJsonPtr)["output_columns"].AppendValue(t.Token());
            }
            (*plainJsonPtr)["output_columns"].AppendValue(ToString(EColumn::Label));
        });

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
        .AddLongOption("boosting-type")
        .RequiredArgument("BoostingType")
        .Help("Set boosting type (Dynamic, Plain). By default CatBoost use dynamic-boosting scheme. For best performance you could set it to Plain.")
        .Handler1T<TString>([plainJsonPtr](const TString& boostingType) {
            (*plainJsonPtr)["boosting_type"] = boostingType;
        });

    parser
            .AddLongOption("data-partition")
            .RequiredArgument("PartitionType")
            .Help("Sets method to split learn samples between multiple workers (GPU only currently). Posible values FeatureParallel, DocParallel. Default depends on learning mode and dataset.")
            .Handler1T<TString>([plainJsonPtr](const TString& type) {
                (*plainJsonPtr)["data_partition"] = type;
            });

    parser.AddLongOption("od-pval",
                         "set threshold for overfitting detector and stop matrixnet automaticaly. For good results use threshold in [1e-10, 1e-2]. Requires any test part.")
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
        .Handler1T<TString>([plainJsonPtr](const TString& type) {
            (*plainJsonPtr)["od_type"] = type;
        });

    //tree options
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

    parser.AddLongOption('n', "depth", "tree depth")
        .RequiredArgument("int")
        .Handler1T<int>([plainJsonPtr](int depth) {
            (*plainJsonPtr)["depth"] = depth;
        });

    parser.AddLongOption("l2-leaf-reg", "Regularization value. Should be >= 0")
        .RequiredArgument("float")
        .Handler1T<float>([plainJsonPtr](float reg) {
            (*plainJsonPtr)["l2_leaf_reg"] = reg;
        });

    parser.AddLongOption("model-size-reg", "Model size regularization coefficient. Should be >= 0")
         .RequiredArgument("float")
         .Handler1T<float>([plainJsonPtr](float reg) {
             (*plainJsonPtr)["model_size_reg"] = reg;
         });

    parser.AddLongOption("random-strength")
        .RequiredArgument("float")
        .Handler1T<float>([plainJsonPtr](float randomStrength) {
            (*plainJsonPtr)["random_strength"] = randomStrength;
        })
        .Help("score stdandart deviation multiplier");

    parser.AddLongOption("leaf-estimation-method", "One of {Newton, Gradient}")
        .RequiredArgument("method-name")
        .Handler1T<TString>([plainJsonPtr](const TString& method) {
            (*plainJsonPtr)["leaf_estimation_method"] = method;
        });

    parser
        .AddLongOption("score-function")
        .RequiredArgument("STRING")
        .Help("Could be change during GPU learning only. Change score function to use. One of {Correlation, SolarL2}")
        .Handler1T<TString>([plainJsonPtr](const TString& func) {
            (*plainJsonPtr)["score_function"] = func;
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

    parser
        .AddLongOption("bootstrap-type")
        .RequiredArgument("STRING")
        .Help("Bootstrap type. Change default way of sampling documents weights. One of"
              " Poisson,"
              " Bayesian,"
              " Bernoulli,"
              " No. By default CatBoost uses bayesian bootstrap type")
        .Handler1T<TString>([plainJsonPtr](const TString& type) {
            (*plainJsonPtr)["bootstrap_type"] = type;
        });

    parser.AddLongOption("bagging-temperature")
        .AddLongName("tmp")
        .RequiredArgument("float")
        .Handler1T<float>([plainJsonPtr](float baggingTemperature) {
            (*plainJsonPtr)["bagging_temperature"] = baggingTemperature;
        })
        .Help("Controls intensity of Bayesian bagging. The higher the temperature the more aggressive bagging is. Typical values are in range [0, 1] (0 - no bagging, 1 - default). Available for Bayesian bootstap only");

    parser.AddLongOption("sampling-frequency")
        .RequiredArgument("string")
        .Handler1T<TString>([plainJsonPtr](const TString& target) {
            (*plainJsonPtr)["sampling_frequency"] = target;
        })
        .Help("Controls how frequently to sample weights and objects when constructing trees. Possible values are PerTree and PerTreeLevel.");

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

    parser.AddLongOption("max-ctr-complexity", "max count of cat features for combinations ctr")
        .RequiredArgument("int")
        .Handler1T<int>([plainJsonPtr](int count) {
            (*plainJsonPtr).InsertValue("max_ctr_complexity", count);
        });

    parser.AddLongOption("simple-ctr",
                         "Ctr description should be written in format CtrType[:TargetBorderCount=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num/denum]. CtrType should be one of: Borders, Buckets, BinarizedTargetMeanValue, Counter. TargetBorderType should be one of: Median, GreedyLogSum, UniformAndQuantiles, MinEntropy, MaxLogSum, Uniform")
        .RequiredArgument("comma separated list of ctr descriptions")
        .Handler1T<TString>([plainJsonPtr](const TString& ctrDescriptionLine) {
            for (const auto& oneCtrConfig : StringSplitter(ctrDescriptionLine).Split(',')) {
                (*plainJsonPtr)["simple_ctr"].AppendValue(oneCtrConfig.Token());
            }
        });

    parser.AddLongOption("combinations-ctr",
                         "Ctr description should be written in format CtrType[:TargetBorderCount=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num/denum]. CtrType should be one of: Borders, Buckets, BinarizedTargetMeanValue, Counter. TargetBorderType should be one of: Median, GreedyLogSum, UniformAndQuantiles, MinEntropy, MaxLogSum, Uniform")
        .RequiredArgument("comma separated list of ctr descriptions")
        .Handler1T<TString>([plainJsonPtr](const TString& ctrDescriptionLine) {
            for (const auto& oneCtrConfig : StringSplitter(ctrDescriptionLine).Split(',')) {
                (*plainJsonPtr)["combinations_ctr"].AppendValue(oneCtrConfig.Token());
            }
        });

    parser.AddLongOption("per-feature-ctr",
                         "Ctr description should be written in format FeatureId:CtrType:[:TargetBorderCount=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num/denum]. CtrType should be one of: Borders, Buckets, BinarizedTargetMeanValue, Counter. TargetBorderType should be one of: Median, GreedyLogSum, UniformAndQuantiles, MinEntropy, MaxLogSum, Uniform")
        .AddLongName("feature-ctr")
        .RequiredArgument("comma separated list of ctr descriptions")
        .Handler1T<TString>([plainJsonPtr](const TString& ctrDescriptionLine) {
            for (const auto& oneCtrConfig : StringSplitter(ctrDescriptionLine).Split(';')) {
                (*plainJsonPtr)["per_feature_ctr"].AppendValue(oneCtrConfig.Token());
            }
        });

    //legacy fallback
    parser.AddLongOption("ctr",
                         "Ctr description should be written in format FeatureId:CtrType:[:TargetBorderCount=BorderCount][:TargetBorderType=BorderType][:CtrBorderCount=Count][:CtrBorderType=Type][:Prior=num/denum]. CtrType should be one of: Borders, Buckets, BinarizedTargetMeanValue, Counter. TargetBorderType should be one of: Median, GreedyLogSum, UniformAndQuantiles, MinEntropy, MaxLogSum, Uniform")
        .RequiredArgument("comma separated list of ctr descriptions")
        .Handler1T<TString>([plainJsonPtr](const TString& ctrDescriptionLine) {
            for (const auto& oneCtrConfig : StringSplitter(ctrDescriptionLine).Split(',')) {
                (*plainJsonPtr)["ctr_description"].AppendValue(oneCtrConfig.Token());
            }
        });

    parser.AddLongOption("counter-calc-method", "Should be one of {Full, SkipTest}")
        .RequiredArgument("method-name")
        .Handler1T<TString>([plainJsonPtr](const TString& method) {
            (*plainJsonPtr).InsertValue("counter_calc_method", method);
        });

    parser.AddLongOption("ctr-leaf-count-limit",
                         "Limit maximum ctr leaf count. If there are more leafs than limit, it'll select top values by frequency and put the rest into trashbucket. This option reduces resulting model size and amount of memory used during training. But it might affect the resulting quality. CPU only")
        .RequiredArgument("maxLeafCount")
        .Handler1T<ui64>([plainJsonPtr](ui64 maxLeafCount) {
            (*plainJsonPtr).InsertValue("ctr_leaf_count_limit", maxLeafCount);
        });

    parser.AddLongOption("store-all-simple-ctr",
                         "Do not limit simple ctr leafs count to topN, store all values from learn set")
        .NoArgument()
        .Handler0([plainJsonPtr]() {
            (*plainJsonPtr).InsertValue("store_all_simple_ctr", true);
        });

    parser.AddLongOption("model-format")
        .RequiredArgument("comma separated list of formats")
        .Handler1T<TString>([plainJsonPtr](const TString& formatsLine) {
            for (const auto& format : StringSplitter(formatsLine).Split(',')) {
                (*plainJsonPtr)["model_format"].AppendValue(format.Token());
            }
        })
        .Help("Alters format of output file for the model. Supported values {CatboostBinary, AppleCoreML, CPP, Python}. Default is CatboostBinary. Corresponding extensions will be added to model-file if more than one format is set.");

    parser.AddLongOption("one-hot-max-size")
        .RequiredArgument("size_t")
        .Handler1T<size_t>([plainJsonPtr](const size_t oneHotMaxSize) {
            (*plainJsonPtr).InsertValue("one_hot_max_size", oneHotMaxSize);
        })
        .Help("If parameter is specified than features with no more than specified value different values will be converted to float features using one-hot encoding. No ctrs will be calculated on this features.");

    //data processing
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

    parser.AddLongOption("classes-count", "number of classes")
        .RequiredArgument("int")
        .Handler1T<int>([plainJsonPtr](const int classesCount) {
            (*plainJsonPtr).InsertValue("classes_count", classesCount);
        })
        .Help("Takes effect only with MutliClass loss function. If classes-count is given (and class-names is not given), then each class label should be less than that number.");

    parser.AddLongOption("class-names", "names for classes.")
        .RequiredArgument("comma separated list of names")
        .Handler1T<TString>([plainJsonPtr](const TString& namesLine) {
            for (const auto& t : StringSplitter(namesLine).Split(',')) {
                (*plainJsonPtr)["class_names"].AppendValue(t.Token());
            }
        })
        .Help("Takes effect only with MutliClass/LogLoss loss functions. Wihout this parameter classes are 0, 1, ..., classes-count - 1");

    parser.AddLongOption("class-weights", "Weights for classes.")
        .RequiredArgument("comma separated list of weights")
        .Handler1T<TString>([plainJsonPtr](const TString& weightsLine) {
            for (const auto& t : StringSplitter(weightsLine).Split(',')) {
                (*plainJsonPtr)["class_weights"].AppendValue(FromString<float>(t.Token()));
            }
        })
        .Help("Takes effect only with MutliClass/LogLoss loss functions. Number of classes indicated by classes-count, class-names and class-weights should be the same");

    parser.AddLongOption('x', "border-count", "count of borders per float feature. Should be in range [1, 255]")
        .RequiredArgument("int")
        .Handler1T<int>([plainJsonPtr](int count) {
            (*plainJsonPtr)["border_count"] = count;
        });

    parser.AddLongOption("feature-border-type",
                         "Should be one of: Median, GreedyLogSum, UniformAndQuantiles, MinEntropy, MaxLogSum")
        .RequiredArgument("border-type")
        .Handler1T<TString>([plainJsonPtr](const TString& type) {
            (*plainJsonPtr)["feature_border_type"] =
                type;
        });

    parser.AddLongOption("nan-mode", "Should be one of: {Min, Max, Forbidden}. Default: Min")
        .RequiredArgument("nan-mode")
        .Handler1T<TString>([plainJsonPtr](const TString& nanMode) {
            (*plainJsonPtr)["nan_mode"] =
                nanMode;
        });

    parser.AddCharOption('T', "worker thread count (default: core count)")
        .AddLongName("thread-count")
        .RequiredArgument("count")
        .Handler1T<int>([plainJsonPtr](int count) {
            (*plainJsonPtr).InsertValue("thread_count", count);
        });

    parser.AddLongOption("used-ram-limit", "Try to limit used memory. CPU only. WARNING: This option affects CTR memory usage only.\nAllowed suffixes: GB, MB, KB in different cases")
            .RequiredArgument("TARGET_RSS")
            .Handler1T<TString>([&plainJsonPtr](const TString& param) {
                (*plainJsonPtr)["used_ram_limit"] = ParseMemorySizeDescription(param);
            });

    parser.AddLongOption("allow-writing-files", "Allow writing files on disc. Possible values: true, false")
            .RequiredArgument("bool")
            .Handler1T<TString>([&plainJsonPtr](const TString& param) {
                (*plainJsonPtr)["allow_writing_files"] = FromString<bool>(param);
            });

    parser
            .AddLongOption("gpu-ram-part")
            .RequiredArgument("double")
            .Help("Part of gpu ram to use")
            .Handler1T<double>([&plainJsonPtr](const double part) {
                (*plainJsonPtr)["gpu_ram_part"] = part;
            });

    parser
            .AddLongOption("pinned-memory-size")
            .RequiredArgument("int")
            .Help("GPU only. Minimum CPU pinned memory to use")
            .Handler1T<TString>([&plainJsonPtr](const TString& param) {
                (*plainJsonPtr)["pinned_memory_size"] = ParseMemorySizeDescription(param);
            });

    parser
        .AddLongOption("gpu-cat-features-storage")
        .RequiredArgument("String")
        .Help("GPU only. One of GpuRam, CpuPinnedMemory. Default GpuRam")
        .Handler1T<TString>([plainJsonPtr](const TString& storage) {
            (*plainJsonPtr)["gpu_cat_features_storage"] = storage;
        });

    parser
        .AddLongOption("task-type")
        .RequiredArgument("String")
        .Help("One of CPU, GPU")
        .Handler1T<TString>([plainJsonPtr](const TString& taskType) {
            (*plainJsonPtr)["task_type"] = taskType;
        });

    parser
        .AddLongOption("devices")
        .RequiredArgument("String")
        .Help("List of devices. Could be enumeration with : separator (1:2:4), range 1-3; 1-3:5. Default -1 (use all devices)")
        .Handler1T<TString>([plainJsonPtr](const TString& devices) {
            (*plainJsonPtr)["devices"] = devices;
        });

    parser
        .AddLongOption("node-type")
        .RequiredArgument("String")
        .Help("One of Master, Worker, SingleHost; default is SingleHost")
        .Handler1T<TString>([plainJsonPtr](const TString& nodeType) {
            (*plainJsonPtr)["node_type"] = nodeType;
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

    parser.AddLongOption('r', "seed")
        .AddLongName("random-seed")
        .RequiredArgument("count")
        .Handler1T<ui64>([plainJsonPtr](ui64 seed) {
            (*plainJsonPtr)["random_seed"] = seed;
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

    parser.AddLongOption("params-file", "Path to JSON file with params.")
        .RequiredArgument("PATH")
        .StoreResult(paramsPath)
        .Help("If param is given in json file and in command line then one from command line will be used.");


    parser.SetFreeArgsNum(0);
    NLastGetopt::TOptsParseResult parserResult{&parser, argc, argv};
}
